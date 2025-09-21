from typing import List, Dict, Any
from config import get_config, get_c2_config
from base_retrieval import BaseRetrieval

class C2Retrieval(BaseRetrieval):
    """C2 태스크: Glass Type + 재료 매칭 기반 칵테일 추천 검색"""
    
    def __init__(self, use_python_config: bool = True):
        """Initialize C2 retrieval system"""
        if use_python_config:
            # Python 설정 사용
            config = get_config()
            c2_config = get_c2_config()
        else:
            # 기존 JSON 설정 사용 (하위 호환성)
            import json
            with open("config.json", 'r') as f:
                config = json.load(f)
            c2_config = config['c2_config']
        
        # 기본 클래스 초기화
        super().__init__(config, c2_config)
        self.c2_config = c2_config  # 편의를 위해 별도 저장
    
    def extract_cocktail_keywords(self, user_question: str) -> Dict[str, List[str]]:
        """LLM을 사용하여 사용자 질문에서 키워드 추출"""
        result = super().extract_keywords(user_question)
        
        # C2 특화: 기본값 설정
        if 'cocktail' not in result:
            result['cocktail'] = []
        if 'include_ingredients' not in result:
            result['include_ingredients'] = []
        if 'exclude_ingredients' not in result:
            result['exclude_ingredients'] = []
        if 'glassType' not in result:
            result['glassType'] = []
        if 'category' not in result:
            result['category'] = []
            
        return result
    
    def find_initial_cocktails_by_name(self, cocktail_keywords: List[str]) -> List[Dict[str, Any]]:
        """칵테일 키워드를 임베딩하여 name_embedding과 유사도 비교로 top-3 선정"""
        if not cocktail_keywords:
            return []
        
        # 모든 칵테일 키워드를 하나의 문자열로 결합
        combined_keywords = " ".join(cocktail_keywords)
        query_embedding = self.get_embedding(combined_keywords)
        
        cocktail_similarities = []
        
        with self.driver.session() as session:
            # 모든 칵테일의 name_embedding 가져오기
            query = """
            MATCH (c:Cocktail)
            WHERE c.name_embedding IS NOT NULL
            RETURN c.name as name, c.name_embedding as embedding
            """
            
            result = session.run(query)
            
            for record in result:
                if record['embedding']:
                    # 코사인 유사도 계산
                    cocktail_embedding = record['embedding']
                    similarity = self.calculate_cosine_similarity(query_embedding, cocktail_embedding)
                    cocktail_similarities.append({
                        'name': record['name'],
                        'similarity': similarity
                    })
        
        # 유사도 기준으로 정렬하고 상위 3개 선택
        cocktail_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_3 = cocktail_similarities[:self.c2_config['initial_top_k']]
        
        print(f"   → 칵테일 키워드 '{combined_keywords}'로 선정된 상위 3개:")
        for i, cocktail in enumerate(top_3, 1):
            print(f"      {i}. {cocktail['name']} (유사도: {cocktail['similarity']:.3f})")
        
        return top_3
    
    def find_cocktails_by_node_similarities(self, cocktail_keywords: List[str], 
                                          include_ingredients: List[str], 
                                          glass_types: List[str], 
                                          categories: List[str]) -> List[Dict[str, Any]]:
        """각 노드 타입별 name_embedding 유사도로 칵테일 찾기"""
        
        candidate_cocktails = {}  # {cocktail_name: total_score}
        
        with self.driver.session() as session:
            # 1. Cocktail 노드에서 직접 검색 (cocktail_keywords가 있는 경우)
            if cocktail_keywords:
                print(f"   → Cocktail 노드 검색: {cocktail_keywords}")
                for keyword in cocktail_keywords:
                    cocktail_results = self.find_similar_cocktails_by_name_embedding(keyword, session)
                    for cocktail_name, score in cocktail_results:
                        candidate_cocktails[cocktail_name] = candidate_cocktails.get(cocktail_name, 0) + score
            
            # 2. Ingredient 노드 → 연결된 Cocktail (모든 include_ingredients를 동시에 가진 칵테일만)
            if include_ingredients:
                print(f"   → Ingredient 노드 검색: {include_ingredients}")
                # 각 ingredient_keyword에 대해 가장 유사한 실제 ingredient 찾기
                matched_ingredients = []
                for ingredient_keyword in include_ingredients:
                    similar_ingredients = self.find_similar_ingredients_by_name_embedding(ingredient_keyword, session)
                    if similar_ingredients:
                        best_ingredient, best_score = similar_ingredients[0]  # 가장 유사한 것만
                        matched_ingredients.append((best_ingredient, best_score))
                        print(f"      '{ingredient_keyword}' → '{best_ingredient}' (유사도: {best_score:.3f})")
                
                if matched_ingredients:
                    # 모든 matched_ingredients를 동시에 가진 칵테일 찾기
                    ingredient_names = [ing[0] for ing in matched_ingredients]
                    avg_score = sum(score for _, score in matched_ingredients) / len(matched_ingredients)
                    
                    cocktails_with_all_ingredients = self.find_cocktails_with_all_ingredients(ingredient_names)
                    print(f"      모든 재료 {ingredient_names}를 가진 칵테일: {len(cocktails_with_all_ingredients)}개")
                    
                    for cocktail_name in cocktails_with_all_ingredients:
                        candidate_cocktails[cocktail_name] = candidate_cocktails.get(cocktail_name, 0) + avg_score
            
            # 3. GlassType 노드 → 연결된 Cocktail
            if glass_types:
                print(f"   → GlassType 노드 검색: {glass_types}")
                for glass_keyword in glass_types:
                    # 가장 유사한 GlassType 노드 찾기
                    similar_glass_types = self.find_similar_glasstypes_by_name_embedding(glass_keyword, session)
                    for glass_name, glass_score in similar_glass_types:
                        # 해당 GlassType과 연결된 Cocktail들 찾기
                        cocktail_results = self.find_cocktails_by_glasstype(glass_name, session)
                        for cocktail_name in cocktail_results:
                            candidate_cocktails[cocktail_name] = candidate_cocktails.get(cocktail_name, 0) + glass_score
            
            # 4. Category 노드 → 연결된 Cocktail
            if categories:
                print(f"   → Category 노드 검색: {categories}")
                for category_keyword in categories:
                    # 가장 유사한 Category 노드 찾기
                    similar_categories = self.find_similar_categories_by_name_embedding(category_keyword, session)
                    for category_name, cat_score in similar_categories:
                        # 해당 Category와 연결된 Cocktail들 찾기
                        cocktail_results = self.find_cocktails_by_category(category_name, session)
                        for cocktail_name in cocktail_results:
                            candidate_cocktails[cocktail_name] = candidate_cocktails.get(cocktail_name, 0) + cat_score
        
        # 점수 기준으로 정렬하고 상위 3개 선택
        sorted_cocktails = sorted(candidate_cocktails.items(), key=lambda x: x[1], reverse=True)
        top_cocktails = sorted_cocktails[:self.c2_config['initial_top_k']]
        
        # 결과 포맷 맞추기
        result_cocktails = []
        for cocktail_name, total_score in top_cocktails:
            result_cocktails.append({
                'name': cocktail_name,
                'similarity': total_score
            })
        
        print(f"   → 노드별 유사도 기반 선정된 칵테일:")
        for i, cocktail in enumerate(result_cocktails, 1):
            print(f"      {i}. {cocktail['name']} (총 점수: {cocktail['similarity']:.3f})")
        
        return result_cocktails
    
    def find_similar_cocktails_by_name_embedding(self, keyword: str, session, top_k: int = 3):
        """Cocktail 노드의 name_embedding과 유사도 비교"""
        keyword_embedding = self.get_embedding(keyword)
        
        query = """
        MATCH (c:Cocktail)
        WHERE c.name_embedding IS NOT NULL
        RETURN c.name as name, c.name_embedding as embedding
        """
        result = session.run(query)
        
        similarities = []
        for record in result:
            if record['embedding']:
                cocktail_embedding = record['embedding']
                similarity = self.calculate_cosine_similarity(keyword_embedding, cocktail_embedding)
                similarities.append((record['name'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_ingredients_by_name_embedding(self, keyword: str, session, top_k: int = 3):
        """Ingredient 노드의 name_embedding과 유사도 비교"""
        keyword_embedding = self.get_embedding(keyword)
        
        query = """
        MATCH (i:Ingredient)
        WHERE i.name_embedding IS NOT NULL
        RETURN i.name as name, i.name_embedding as embedding
        """
        result = session.run(query)
        
        similarities = []
        for record in result:
            if record['embedding']:
                ingredient_embedding = record['embedding']
                similarity = self.calculate_cosine_similarity(keyword_embedding, ingredient_embedding)
                similarities.append((record['name'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_glasstypes_by_name_embedding(self, keyword: str, session, top_k: int = 3):
        """GlassType 노드의 name_embedding과 유사도 비교"""
        keyword_embedding = self.get_embedding(keyword)
        
        query = """
        MATCH (g:GlassType)
        WHERE g.name_embedding IS NOT NULL
        RETURN g.name as name, g.name_embedding as embedding
        """
        result = session.run(query)
        
        similarities = []
        for record in result:
            if record['embedding']:
                glass_embedding = record['embedding']
                similarity = self.calculate_cosine_similarity(keyword_embedding, glass_embedding)
                similarities.append((record['name'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_categories_by_name_embedding(self, keyword: str, session, top_k: int = 3):
        """Category 노드의 name_embedding과 유사도 비교"""
        keyword_embedding = self.get_embedding(keyword)
        
        query = """
        MATCH (cat:Category)
        WHERE cat.name_embedding IS NOT NULL
        RETURN cat.name as name, cat.name_embedding as embedding
        """
        result = session.run(query)
        
        similarities = []
        for record in result:
            if record['embedding']:
                category_embedding = record['embedding']
                similarity = self.calculate_cosine_similarity(keyword_embedding, category_embedding)
                similarities.append((record['name'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_cocktails_by_ingredient(self, ingredient_name: str, session):
        """특정 Ingredient와 연결된 Cocktail들 찾기"""
        query = """
        MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient {name: $ingredient})
        RETURN c.name as name
        """
        result = session.run(query, {'ingredient': ingredient_name})
        return [record['name'] for record in result]
    
    def find_cocktails_with_all_ingredients(self, ingredient_names: List[str]):
        """모든 ingredients를 동시에 가진 칵테일들 찾기 (AND 조건)"""
        if not ingredient_names:
            return []
        
        with self.driver.session() as session:
            # 동적으로 MATCH 패턴 생성
            match_patterns = []
            where_conditions = []
            
            for i, ingredient_name in enumerate(ingredient_names):
                match_patterns.append(f"(c)-[:HAS_INGREDIENT]->(i{i}:Ingredient)")
                where_conditions.append(f"i{i}.name = $ingredient_{i}")
            
            query = f"""
            MATCH (c:Cocktail), {", ".join(match_patterns)}
            WHERE {" AND ".join(where_conditions)}
            RETURN DISTINCT c.name as name
            """
            
            # 파라미터 딕셔너리 생성
            params = {f'ingredient_{i}': name for i, name in enumerate(ingredient_names)}
            
            result = session.run(query, params)
            return [record['name'] for record in result]
    
    def find_cocktails_by_glasstype(self, glass_name: str, session):
        """특정 GlassType과 연결된 Cocktail들 찾기"""
        query = """
        MATCH (c:Cocktail)-[:HAS_GLASSTYPE]->(g:GlassType {name: $glassType})
        RETURN c.name as name
        """
        result = session.run(query, {'glassType': glass_name})
        return [record['name'] for record in result]
    
    def find_cocktails_by_category(self, category_name: str, session):
        """특정 Category와 연결된 Cocktail들 찾기"""
        query = """
        MATCH (c:Cocktail)-[:CATEGORY]->(cat:Category {name: $category})
        RETURN c.name as name
        """
        result = session.run(query, {'category': category_name})
        return [record['name'] for record in result]
    
    def find_common_attributes(self, top_cocktails: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """상위 3개 칵테일이 공통으로 가리키는 category, glasstype, ingredient 찾기"""
        if not top_cocktails:
            return {"categories": [], "glass_types": [], "ingredients": []}
        
        cocktail_names = [cocktail['name'] for cocktail in top_cocktails]
        
        with self.driver.session() as session:
            # 각 칵테일의 category, glasstype, ingredient 수집
            all_categories = []
            all_glass_types = []
            all_ingredients = []
            
            for name in cocktail_names:
                query = """
                MATCH (c:Cocktail {name: $name})
                OPTIONAL MATCH (c)-[:CATEGORY]->(cat:Category)
                OPTIONAL MATCH (c)-[:HAS_GLASSTYPE]->(g:GlassType)
                OPTIONAL MATCH (c)-[:HAS_INGREDIENT]->(i:Ingredient)
                RETURN cat.name as category, g.name as glassType, collect(DISTINCT i.name) as ingredients
                """
                
                result = session.run(query, {'name': name})
                record = result.single()
                
                if record:
                    if record['category']:
                        all_categories.append(record['category'])
                    if record['glassType']:
                        all_glass_types.append(record['glassType'])
                    if record['ingredients']:
                        all_ingredients.extend(record['ingredients'])
            
            # 공통 속성 찾기 (모든 칵테일이 공유하는 것들)
            common_categories = []
            common_glass_types = []
            common_ingredients = []
            
            # Category - 모든 칵테일이 같은 카테고리를 가지는지 확인
            if len(set(all_categories)) == 1 and all_categories:
                common_categories = list(set(all_categories))
            
            # Glass Type - 모든 칵테일이 같은 글라스 타입을 가지는지 확인
            if len(set(all_glass_types)) == 1 and all_glass_types:
                common_glass_types = list(set(all_glass_types))
            
            # Ingredients - 모든 칵테일이 공통으로 가지는 재료 찾기
            ingredient_counts = {}
            for ingredient in all_ingredients:
                ingredient_counts[ingredient] = ingredient_counts.get(ingredient, 0) + 1
            
            # 모든 칵테일에 나타나는 재료만 선택
            num_cocktails = len(cocktail_names)
            common_ingredients = [ingredient for ingredient, count in ingredient_counts.items() 
                                if count == num_cocktails]
            
            print(f"   → 공통 속성:")
            print(f"      카테고리: {common_categories}")
            print(f"      글라스 타입: {common_glass_types}")
            print(f"      공통 재료: {common_ingredients}")
            
            return {
                "categories": common_categories,
                "glass_types": common_glass_types,
                "ingredients": common_ingredients
            }
    
    def expand_by_common_attributes(self, common_attributes: Dict[str, List[str]]) -> List[str]:
        """공통 속성을 가진 칵테일들 검색"""
        candidate_cocktails = []
        
        with self.driver.session() as session:
            # Category로 검색
            for category in common_attributes.get("categories", []):
                query = """
                MATCH (c:Cocktail)-[:CATEGORY]->(cat:Category {name: $category})
                RETURN c.name as name
                """
                result = session.run(query, {'category': category})
                for record in result:
                    if record['name'] not in candidate_cocktails:
                        candidate_cocktails.append(record['name'])
            
            # Glass Type으로 검색
            for glass_type in common_attributes.get("glass_types", []):
                query = """
                MATCH (c:Cocktail)-[:HAS_GLASSTYPE]->(g:GlassType {name: $glassType})
                RETURN c.name as name
                """
                result = session.run(query, {'glassType': glass_type})
                for record in result:
                    if record['name'] not in candidate_cocktails:
                        candidate_cocktails.append(record['name'])
            
            # Common Ingredient로 검색
            for ingredient in common_attributes.get("ingredients", []):
                query = """
                MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient {name: $ingredient})
                RETURN c.name as name
                """
                result = session.run(query, {'ingredient': ingredient})
                for record in result:
                    if record['name'] not in candidate_cocktails:
                        candidate_cocktails.append(record['name'])
        
        print(f"   → 공통 속성으로 확장된 칵테일: {len(candidate_cocktails)}개")
        print(f"      {candidate_cocktails}")
        
        return candidate_cocktails
    
    def filter_by_ingredient_overlap(self, candidate_cocktails: List[str], 
                                   highest_similarity_cocktail: str) -> List[str]:
        """가장 유사도가 높았던 칵테일과 재료 겹치는 개수가 많은 순으로 필터링"""
        if not candidate_cocktails or not highest_similarity_cocktail:
            return candidate_cocktails
        
        with self.driver.session() as session:
            # 가장 유사도가 높았던 칵테일의 재료 가져오기
            query = """
            MATCH (c:Cocktail {name: $name})-[:HAS_INGREDIENT]->(i:Ingredient)
            RETURN collect(i.name) as ingredients
            """
            result = session.run(query, {'name': highest_similarity_cocktail})
            record = result.single()
            
            if not record or not record['ingredients']:
                return candidate_cocktails
            
            reference_ingredients = set(record['ingredients'])
            print(f"   → 기준 칵테일 '{highest_similarity_cocktail}'의 재료: {list(reference_ingredients)}")
            
            # 각 후보 칵테일의 재료 겹치는 개수 계산
            cocktail_overlaps = []
            
            for cocktail_name in candidate_cocktails:
                if cocktail_name == highest_similarity_cocktail:
                    continue  # 기준 칵테일 자체는 제외
                
                query = """
                MATCH (c:Cocktail {name: $name})-[:HAS_INGREDIENT]->(i:Ingredient)
                RETURN collect(i.name) as ingredients
                """
                result = session.run(query, {'name': cocktail_name})
                record = result.single()
                
                if record and record['ingredients']:
                    candidate_ingredients = set(record['ingredients'])
                    overlap_count = len(reference_ingredients.intersection(candidate_ingredients))
                    cocktail_overlaps.append({
                        'name': cocktail_name,
                        'overlap_count': overlap_count
                    })
            
            # 재료 겹치는 개수 기준으로 정렬
            cocktail_overlaps.sort(key=lambda x: x['overlap_count'], reverse=True)
            
            # 최대 final_top_k개까지 선택
            final_cocktails = [item['name'] for item in cocktail_overlaps[:self.c2_config['final_top_k']]]
            
            print(f"   → 재료 겹치는 개수 기준 상위 {len(final_cocktails)}개 선정:")
            for i, item in enumerate(cocktail_overlaps[:self.c2_config['final_top_k']], 1):
                print(f"      {i}. {item['name']} (겹치는 재료: {item['overlap_count']}개)")
            
            return final_cocktails
    
    def get_cocktail_details(self, cocktail_names: List[str]) -> List[Dict[str, Any]]:
        """칵테일 상세 정보 가져오기"""
        cocktails = []
        with self.driver.session() as session:
            for name in cocktail_names:
                query = """
                MATCH (c:Cocktail {name: $name})
                OPTIONAL MATCH (c)-[:CATEGORY]->(cat:Category)
                OPTIONAL MATCH (c)-[:HAS_GLASSTYPE]->(g:GlassType)
                OPTIONAL MATCH (c)-[:HAS_INGREDIENT]->(i:Ingredient)
                RETURN c.name as name, 
                       c.description as description,
                       c.instructions as instructions,
                       c.imageDescription as imageDescription,
                       c.alcoholic as alcoholic,
                       cat.name as category,
                       g.name as glassType,
                       collect(DISTINCT i.name) as ingredients
                """
                
                result = session.run(query, {'name': name})
                record = result.single()
                
                if record:
                    cocktails.append({
                        'name': record['name'],
                        'description': record['description'],
                        'instructions': record['instructions'],
                        'imageDescription': record['imageDescription'],
                        'alcoholic': record['alcoholic'],
                        'category': record['category'],
                        'glassType': record['glassType'],
                        'ingredients': record['ingredients']
                    })
        
        return cocktails
    
    def find_best_glass_type(self, glass_keywords: List[str]) -> str:
        """Glass Type 키워드에서 가장 유사한 실제 GlassType 찾기"""
        with self.driver.session() as session:
            for keyword in glass_keywords:
                similar_glass_types = self.find_similar_glasstypes_by_name_embedding(keyword, session, top_k=1)
                if similar_glass_types:
                    best_glass, score = similar_glass_types[0]
                    print(f"   → '{keyword}' → '{best_glass}' (유사도: {score:.3f})")
                    return best_glass
        return None
    
    def infer_glass_type_from_context(self, cocktail_keywords: List[str], categories: List[str]) -> str:
        """Cocktail이나 Category에서 글라스 타입 추정"""
        with self.driver.session() as session:
            # Cocktail 키워드가 있으면 해당 칵테일의 글라스 타입 참조
            if cocktail_keywords:
                for keyword in cocktail_keywords:
                    similar_cocktails = self.find_similar_cocktails_by_name_embedding(keyword, session, top_k=1)
                    if similar_cocktails:
                        cocktail_name, _ = similar_cocktails[0]
                        # 해당 칵테일의 글라스 타입 가져오기
                        query = """
                        MATCH (c:Cocktail {name: $name})-[:HAS_GLASSTYPE]->(g:GlassType)
                        RETURN g.name as glassType
                        """
                        result = session.run(query, {'name': cocktail_name})
                        record = result.single()
                        if record:
                            glass_type = record['glassType']
                            print(f"   → '{keyword}' 칵테일에서 글라스 타입 추정: {glass_type}")
                            return glass_type
            
            # 그래도 없으면 기본값
            print("   → 글라스 타입을 추정할 수 없어 기본값 사용")
            return None
    
    def get_cocktails_by_glass_type(self, glass_type: str) -> List[str]:
        """특정 글라스 타입을 가진 모든 칵테일 반환"""
        with self.driver.session() as session:
            query = """
            MATCH (c:Cocktail)-[:HAS_GLASSTYPE]->(g:GlassType {name: $glassType})
            RETURN c.name as name
            """
            result = session.run(query, {'glassType': glass_type})
            return [record['name'] for record in result]
    
    def score_cocktails_by_ingredient_matching(self, cocktail_names: List[str], include_ingredients: List[str]) -> List[Dict[str, Any]]:
        """칵테일들을 재료 매칭 점수로 평가"""
        with self.driver.session() as session:
            # 각 ingredient keyword에 대해 가장 유사한 실제 ingredient 찾기
            matched_ingredients = []
            for ingredient_keyword in include_ingredients:
                similar_ingredients = self.find_similar_ingredients_by_name_embedding(ingredient_keyword, session)
                if similar_ingredients:
                    best_ingredient, best_score = similar_ingredients[0]
                    matched_ingredients.append(best_ingredient)
                    print(f"   → '{ingredient_keyword}' → '{best_ingredient}'")
            
            if not matched_ingredients:
                return []
            
            # 각 칵테일의 재료 매칭 점수 계산
            scored_cocktails = []
            for cocktail_name in cocktail_names:
                query = """
                MATCH (c:Cocktail {name: $name})-[:HAS_INGREDIENT]->(i:Ingredient)
                RETURN collect(i.name) as ingredients
                """
                result = session.run(query, {'name': cocktail_name})
                record = result.single()
                
                if record and record['ingredients']:
                    cocktail_ingredients = set(record['ingredients'])
                    matched_count = len([ing for ing in matched_ingredients if ing in cocktail_ingredients])
                    # 매칭된 재료 개수 / 전체 요청 재료 개수 = 매칭 비율
                    match_ratio = matched_count / len(matched_ingredients)
                    
                    scored_cocktails.append({
                        'name': cocktail_name,
                        'score': match_ratio,
                        'matched_ingredients': [ing for ing in matched_ingredients if ing in cocktail_ingredients]
                    })
            
            # 점수 기준으로 정렬
            scored_cocktails.sort(key=lambda x: x['score'], reverse=True)
            return scored_cocktails
    
    def score_cocktails_by_name_similarity(self, cocktail_names: List[str], cocktail_keywords: List[str], user_question: str) -> List[Dict[str, Any]]:
        """칵테일들을 이름 유사도로 평가"""
        if cocktail_keywords:
            # Cocktail 키워드가 있으면 name_embedding 유사도
            combined_keywords = " ".join(cocktail_keywords)
            query_embedding = self.get_embedding(combined_keywords)
        else:
            # 전체 질문으로 유사도
            query_embedding = self.get_embedding(user_question)
        
        scored_cocktails = []
        with self.driver.session() as session:
            for cocktail_name in cocktail_names:
                query = """
                MATCH (c:Cocktail {name: $name})
                RETURN c.name_embedding as embedding
                """
                result = session.run(query, {'name': cocktail_name})
                record = result.single()
                
                if record and record['embedding']:
                    cocktail_embedding = record['embedding']
                    similarity = self.calculate_cosine_similarity(query_embedding, cocktail_embedding)
                    scored_cocktails.append({
                        'name': cocktail_name,
                        'score': similarity
                    })
        
        # 점수 기준으로 정렬
        scored_cocktails.sort(key=lambda x: x['score'], reverse=True)
        return scored_cocktails
    
    def progressive_ingredient_search(self, glass_cocktails: List[str], include_ingredients: List[str]) -> Dict[int, List[str]]:
        """계층적 재료 매칭: 모든 재료 → 1개씩 뒤에서 제거하며 후보 확보"""
        if not include_ingredients:
            return {0: glass_cocktails[:self.c2_config['target_candidates']]}
        
        print(f"   → 계층적 재료 매칭 시작: {include_ingredients}")
        
        # 재료 키워드들을 실제 재료명으로 매칭
        matched_ingredients = []
        with self.driver.session() as session:
            for ingredient_keyword in include_ingredients:
                similar_ingredients = self.find_similar_ingredients_by_name_embedding(ingredient_keyword, session, top_k=1)
                if similar_ingredients:
                    best_ingredient, score = similar_ingredients[0]
                    matched_ingredients.append(best_ingredient)
                    print(f"      '{ingredient_keyword}' → '{best_ingredient}' (유사도: {score:.3f})")
        
        if not matched_ingredients:
            print("   → 매칭된 재료가 없어 전체 글라스 칵테일 반환")
            return {0: glass_cocktails[:self.c2_config['target_candidates']]}
        
        candidates_by_level = {}
        total_candidates = []
        
        # Level별로 재료 조합 시도 (뒤에서부터 1개씩 제거)
        for level in range(len(matched_ingredients) + 1):
            if len(total_candidates) >= self.c2_config['target_candidates']:
                break
                
            if level == 0:
                # Level 0: 모든 재료 동시 매칭
                current_ingredients = matched_ingredients
                level_name = "모든재료"
            else:
                # Level 1~: 뒤에서부터 level개만큼 제거
                current_ingredients = matched_ingredients[:-level]
                level_name = f"재료-{level}개"
            
            if not current_ingredients:
                break
            
            print(f"   → Level {level} ({level_name}): {current_ingredients}")
            
            # 현재 재료 조합으로 칵테일 검색 (새 세션 사용)
            level_cocktails = self.find_cocktails_with_all_ingredients(current_ingredients)
            
            # 글라스 타입에 해당하는 칵테일만 필터링
            level_cocktails = [c for c in level_cocktails if c in glass_cocktails]
            
            # 이미 선택된 칵테일 제외
            new_cocktails = [c for c in level_cocktails if c not in total_candidates]
            
            print(f"      → 새로 발견한 칵테일: {len(new_cocktails)}개")
            if new_cocktails:
                candidates_by_level[level] = new_cocktails
                total_candidates.extend(new_cocktails)
                
                print(f"         {new_cocktails}")
            
            # 최소 임계값 확인 (Level 0에서만)
            if level == 0 and len(new_cocktails) > self.c2_config['min_candidates_threshold']:
                print(f"   → Level 0에서 충분한 후보({len(new_cocktails)}개) 확보, 추가 레벨 생략")
                break
        
        print(f"   → 계층적 검색 완료: 총 {len(total_candidates)}개 후보 확보")
        print(f"      레벨별 분포: {[(level, len(cocktails)) for level, cocktails in candidates_by_level.items()]}")
        
        return candidates_by_level
    
    def rank_by_image_similarity_grouped(self, candidates_by_level: Dict[int, List[str]], user_question: str) -> List[str]:
        """레벨별로 imageDescription_embedding 유사도 계산 후 순서대로 정렬"""
        if not candidates_by_level:
            return []
        
        print(f"   → 레벨별 imageDescription 유사도 계산 시작")
        question_embedding = self.get_embedding(user_question)
        
        final_ordered_cocktails = []
        
        # 레벨 순서대로 처리 (0, 1, 2, ...)
        for level in sorted(candidates_by_level.keys()):
            cocktail_names = candidates_by_level[level]
            if not cocktail_names:
                continue
            
            print(f"   → Level {level} 처리: {len(cocktail_names)}개 칵테일")
            
            # 현재 레벨의 칵테일들 유사도 계산
            cocktail_similarities = []
            with self.driver.session() as session:
                for name in cocktail_names:
                    query = """
                    MATCH (c:Cocktail {name: $name})
                    RETURN c.imageDescription_embedding as embedding
                    """
                    result = session.run(query, {'name': name})
                    record = result.single()
                    
                    if record and record['embedding']:
                        cocktail_embedding = record['embedding']
                        similarity = self.calculate_cosine_similarity(question_embedding, cocktail_embedding)
                        cocktail_similarities.append((name, similarity))
                    else:
                        # imageDescription_embedding이 없으면 낮은 점수 할당
                        cocktail_similarities.append((name, 0.0))
            
            # 현재 레벨 내에서 유사도 기준으로 정렬
            cocktail_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 정렬된 순서로 최종 리스트에 추가
            level_ordered = [name for name, similarity in cocktail_similarities]
            final_ordered_cocktails.extend(level_ordered)
            
            print(f"      → Level {level} 유사도 순위:")
            for i, (name, similarity) in enumerate(cocktail_similarities, 1):
                print(f"         {i}. {name} (유사도: {similarity:.3f})")
        
        print(f"   → 최종 순서 정렬 완료: {len(final_ordered_cocktails)}개")
        return final_ordered_cocktails
    
    def retrieve(self, user_question: str) -> List[Dict[str, Any]]:
        """Glass Type + 계층적 재료 매칭 기반 칵테일 검색 알고리즘"""
        print(f"C2 Retrieval (Glass Type + 계층적 재료 매칭): 사용자 질문 - {user_question}")
        
        # 1단계: 키워드 추출
        keywords = self.extract_cocktail_keywords(user_question)
        print(f"1단계 - 키워드 추출: {keywords}")
        
        cocktail_keywords = keywords.get('cocktail', [])
        include_ingredients = keywords.get('include_ingredients', [])
        glass_types = keywords.get('glassType', [])
        categories = keywords.get('category', [])
        
        # 2단계: Glass Type 결정
        target_glass_type = None
        if glass_types:
            target_glass_type = self.find_best_glass_type(glass_types)
        else:
            target_glass_type = self.infer_glass_type_from_context(cocktail_keywords, categories)
        
        if not target_glass_type:
            print("❌ 대상 글라스 타입을 결정할 수 없습니다.")
            return []
        
        print(f"2단계 - 대상 글라스 타입: {target_glass_type}")
        
        # 3단계: 해당 글라스 타입 칵테일들 필터링
        glass_cocktails = self.get_cocktails_by_glass_type(target_glass_type)
        print(f"3단계 - {target_glass_type} 글라스 칵테일: {len(glass_cocktails)}개")
        
        if not glass_cocktails:
            print("❌ 해당 글라스 타입의 칵테일을 찾을 수 없습니다.")
            return []
        
        # 4단계: 계층적 재료 매칭으로 후보 수집
        if include_ingredients:
            print(f"4단계 - 계층적 재료 매칭")
            candidates_by_level = self.progressive_ingredient_search(glass_cocktails, include_ingredients)
        else:
            print(f"4단계 - 재료 없음, 이름 기반 검색")
            # 재료가 없으면 이름 유사도로 후보 선정
            scored_cocktails = self.score_cocktails_by_name_similarity(glass_cocktails, cocktail_keywords, user_question)
            top_candidates = [item['name'] for item in scored_cocktails[:self.c2_config['target_candidates']]]
            candidates_by_level = {0: top_candidates}
        
        # 5단계: 레벨별 imageDescription 유사도 정렬
        print(f"5단계 - 레벨별 imageDescription 유사도 정렬")
        ordered_cocktails = self.rank_by_image_similarity_grouped(candidates_by_level, user_question)
        
        # 6단계: 최종 top-k 선정
        final_cocktail_names = ordered_cocktails[:self.c2_config['final_top_k']]
        print(f"6단계 - 최종 선정: {len(final_cocktail_names)}개")
        for i, name in enumerate(final_cocktail_names, 1):
            print(f"   {i}. {name}")
        
        # 상세 정보 가져오기
        final_results = self.get_cocktail_details(final_cocktail_names)
        print(f"최종 결과: {len(final_results)}개 칵테일")
        
        return final_results
