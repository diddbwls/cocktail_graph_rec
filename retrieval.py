import os
import json
import openai
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import List, Dict

class PromptLoader:
    def __init__(self, prompts_dir="prompts", config_path="config.json"):
        self.prompts_dir = prompts_dir
        self.config_path = config_path
        self.prompts = {}
        self.config = {}
        self.load_config()
        self.load_all_prompts()
    
    def load_config(self):
        """config.json에서 설정 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.config_path} not found. Using default config.")
            self.config = {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1000,
                "language": "ko"
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {self.config_path}: {e}")
            self.config = {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1000,
                "language": "ko"
            }
    
    def load_all_prompts(self):
        """모든 프롬프트 파일을 로딩 (JSON 형태로 통일)"""
        prompt_files = {
            'base_system': 'base_system.json',
            'c1': 'c1_visual_similarity.json',
            'c2': 'c2_taste_profile.json',
            'c3': 'c3_classification.json',
            'c4': 'c4_recipe_ingredients.json',
            'task_classifier': 'task_classifier.json'
        }
        
        for key, filename in prompt_files.items():
            filepath = os.path.join(self.prompts_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                    self.prompts[key] = data
            except FileNotFoundError:
                print(f"Warning: {filepath} not found. Using default prompt.")
                self.prompts[key] = {
                    "system_prompt": "You are a cocktail expert.",
                    "task_prompt": "Provide helpful cocktail information."
                }
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in {filepath}: {e}")
                self.prompts[key] = {
                    "system_prompt": "You are a cocktail expert.",
                    "task_prompt": "Provide helpful cocktail information."
                }
    
    def get_system_prompt(self, task_category: str) -> str:
        """태스크별 시스템 프롬프트 조합 (JSON 형태)"""
        base_data = self.prompts.get('base_system', {})
        task_data = self.prompts.get(task_category.lower(), {})
        
        # JSON에서 프롬프트 추출
        base_prompt = base_data.get('system_prompt', '') if isinstance(base_data, dict) else str(base_data)
        task_prompt = task_data.get('task_prompt', '') if isinstance(task_data, dict) else str(task_data)
        
        if task_prompt:
            return f"{base_prompt}\n\n{task_prompt}"
        else:
            return base_prompt
    
    def get_llm_config(self, task_category: str) -> dict:
        """태스크별 LLM 설정 반환 (config.json 기반)"""
        # config.json에서 기본 설정 가져오기
        base_config = {
            'model': self.config.get('model', 'gpt-4o-mini'),
            'temperature': self.config.get('temperature', 0.7),
            'max_tokens': self.config.get('max_tokens', 1000)
        }
        
        # 태스크별 설정이 있으면 오버라이드
        task_specific = self.config.get('task_specific', {})
        if task_category.lower() in task_specific:
            task_config = task_specific[task_category.lower()]
            base_config.update(task_config)
        
        return base_config
    
    def get_task_classifier_prompt(self) -> dict:
        """태스크 분류 프롬프트 정보 반환 (JSON + config.json 조합)"""
        classifier_data = self.prompts.get('task_classifier', {})
        
        # config.json에서 task_classifier 설정 가져오기
        task_specific = self.config.get('task_specific', {})
        classifier_config = task_specific.get('task_classifier', {})
        
        # 기본 config와 태스크별 config 조합
        result = {
            "system_prompt": classifier_data.get('system_prompt', 'You are a task classifier.'),
            "user_prompt": classifier_data.get('user_prompt', ''),
            "temperature": classifier_config.get('temperature', self.config.get('temperature', 0.1)),
            "max_tokens": classifier_config.get('max_tokens', self.config.get('max_tokens', 200)),
            "model": classifier_config.get('model', self.config.get('model', 'gpt-4o-mini'))
        }
        
        # 기타 데이터 추가
        if 'example_outputs' in classifier_data:
            result['example_outputs'] = classifier_data['example_outputs']
            
        return result

class CocktailRetrieval:
    def __init__(self):
        load_dotenv()
        
        # Neo4j 연결 - max_connection_pool_size 제한
        self.driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')),
            max_connection_pool_size=5  # 연결 풀 크기 제한
        )
        
        # OpenAI 설정
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # 프롬프트 로더 초기화
        self.prompt_loader = PromptLoader()
        
    def get_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩으로 변환"""
        if not text or text.strip() == "":
            return [0.0] * 1536
            
        try:
            embedding_model = self.prompt_loader.config.get('embedding_model', 'text-embedding-3-small')
            response = openai.embeddings.create(
                input=text,
                model=embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return [0.0] * 1536
    
    def c1_visual_similarity(self, query_image_description: str, task_type: str) -> Dict:
        """C1: 시각 인식 & 유사 추천 - imageDescription + instructions 임베딩 혼합"""
        image_embedding = self.get_embedding(query_image_description)
        inst_embedding = self.get_embedding(f"visual cocktail instructions for {query_image_description}")
        
        with self.driver.session() as session:
            # Seed nodes: 이미지 설명 + 제작법 벡터 혼합
            result = session.run("""
                CALL db.index.vector.queryNodes('cocktail_imageDescription_embedding', 8, $image_embedding)
                YIELD node as image_node, score as image_score
                
                WITH image_node, image_score
                CALL db.index.vector.queryNodes('cocktail_instructions_embedding', 8, $inst_embedding)
                YIELD node as inst_node, score as inst_score
                WHERE image_node.id = inst_node.id
                
                // 2홉 관계 정보 추가
                MATCH (image_node)-[:HAS_INGREDIENT]->(ingredient:Ingredient)
                MATCH (image_node)-[:CATEGORY]->(category:Category)
                OPTIONAL MATCH (image_node)-[:HAS_GLASSTYPE]->(glass:GlassType)
                
                WITH image_node, (image_score + inst_score) / 2 as combined_score,
                     collect(ingredient.name) as ingredients,
                     category.name as category,
                     glass.name as glass_type
                
                ORDER BY combined_score DESC
                RETURN image_node.name as name, image_node.imageDescription as image_description, 
                       image_node.description as description,
                       ingredients, category, glass_type, combined_score as score
                LIMIT 5
            """, image_embedding=image_embedding, inst_embedding=inst_embedding)
            
            return self._format_c1_results(result, task_type)
    
    def c2_taste_profile(self, query_text: str, task_type: str) -> Dict:
        """C2: 설명 & 테이스팅 노트 - description + instructions 임베딩 혼합"""
        desc_embedding = self.get_embedding(query_text)
        inst_embedding = self.get_embedding(f"taste profile instructions for {query_text}")
        
        with self.driver.session() as session:
            # Seed nodes: 설명 + 제작법 벡터 혼합
            result = session.run("""
                CALL db.index.vector.queryNodes('cocktail_description_embedding', 5, $desc_embedding)
                YIELD node as desc_node, score as desc_score
                
                WITH desc_node, desc_score
                CALL db.index.vector.queryNodes('cocktail_instructions_embedding', 5, $inst_embedding)
                YIELD node as inst_node, score as inst_score
                WHERE desc_node.id = inst_node.id
                
                // 관계 정보 추가 - measure 속성 포함
                MATCH (desc_node)-[r:HAS_INGREDIENT]->(ingredient:Ingredient)
                
                WITH desc_node, (desc_score + inst_score) / 2 as combined_score,
                     collect({ingredient: ingredient.name, measure: r.measure}) as recipe
                
                ORDER BY combined_score DESC
                RETURN desc_node.name as name, desc_node.description as description, 
                       desc_node.instructions as instructions,
                       recipe, combined_score
                LIMIT 3
            """, desc_embedding=desc_embedding, inst_embedding=inst_embedding)
            
            return self._format_c2_results(result, task_type)
    
    def c3_classification(self, query_text: str, task_type: str) -> Dict:
        """C3: 분류/계통 & 메타데이터 - description + 관계 중심"""
        query_embedding = self.get_embedding(query_text)
        
        with self.driver.session() as session:
            if "family" in task_type.lower():
                # 패밀리 추정: 관계 중심
                result = session.run("""
                    CALL db.index.vector.queryNodes('cocktail_description_embedding', 15, $query_embedding)
                    YIELD node, score
                    
                    MATCH (node)-[:CATEGORY]->(category:Category)
                    MATCH (category)<-[:CATEGORY]-(family_members:Cocktail)
                    
                    WITH category, count(family_members) as family_size, 
                         count(node) as matches, avg(score) as avg_similarity
                    
                    ORDER BY family_size DESC, matches DESC
                    RETURN category.name as category_name, family_size, matches, avg_similarity
                    LIMIT 3
                """, query_embedding=query_embedding)
            else:
                # 일반 분류: 정확한 이름 매칭 + 메타데이터
                result = session.run("""
                    MATCH (c:Cocktail)
                    WHERE toLower(c.name) CONTAINS toLower($query_text)
                    
                    MATCH (c)-[:CATEGORY]->(category:Category)
                    MATCH (c)-[:HAS_INGREDIENT]->(ingredient:Ingredient)
                    OPTIONAL MATCH (c)-[:HAS_GLASSTYPE]->(glass:GlassType)
                    
                    RETURN c.name as name, c.description as description, 
                           category.name as category,
                           collect(ingredient.name) as ingredients,
                           glass.name as glass_type
                    LIMIT 1
                """, query_text=query_text)
            
            return self._format_c3_results(result, task_type)
    
    def c4_recipe_ingredients(self, query_text: str, task_type: str) -> Dict:
        """C4: 재료/레시피 & 치환 - instructions + measure 속성 활용"""
        query_embedding = self.get_embedding(query_text)
        
        with self.driver.session() as session:
            if "recipe" in task_type.lower() or "measure" in task_type.lower():
                # 레시피 정보: measure 속성 중점
                result = session.run("""
                    MATCH (c:Cocktail)
                    WHERE toLower(c.name) CONTAINS toLower($query_text)
                    
                    MATCH (c)-[r:HAS_INGREDIENT]->(ingredient:Ingredient)
                    OPTIONAL MATCH (c)-[:HAS_GLASSTYPE]->(glass:GlassType)
                    
                    RETURN c.name as name, c.instructions as instructions,
                           collect({ingredient: ingredient.name, measure: r.measure}) as detailed_recipe,
                           glass.name as glass_type
                    LIMIT 1
                """, query_text=query_text)
            else:
                # 재료 추정: 제작법 벡터 기반
                result = session.run("""
                    CALL db.index.vector.queryNodes('cocktail_instructions_embedding', 10, $query_embedding)
                    YIELD node, score
                    
                    MATCH (node)-[:HAS_INGREDIENT]->(ingredient:Ingredient)
                    MATCH (ingredient)<-[:HAS_INGREDIENT]-(all_cocktails:Cocktail)
                    
                    WITH ingredient, count(node) as usage_frequency, 
                         avg(score) as avg_similarity,
                         count(all_cocktails) as global_popularity
                    ORDER BY usage_frequency DESC, avg_similarity DESC
                    
                    RETURN ingredient.name as ingredient_name, usage_frequency, 
                           avg_similarity, global_popularity
                    LIMIT 5
                """, query_embedding=query_embedding)
            
            return self._format_c4_results(result, task_type)
    
    def _format_c1_results(self, result, task_type: str) -> Dict:
        """C1 결과 포맷팅"""
        context = {
            "task_type": task_type,
            "embedding_strategy": "imageDescription + instructions mixed",
            "cocktails": []
        }
        
        for record in result:
            context["cocktails"].append({
                "name": record["name"],
                "image_description": record["image_description"],
                "description": record["description"],
                "ingredients": record["ingredients"],
                "category": record["category"],
                "glass_type": record["glass_type"],
                "similarity_score": round(record["score"], 4)
            })
        
        return context
    
    def _format_c2_results(self, result, task_type: str) -> Dict:
        """C2 결과 포맷팅"""
        context = {
            "task_type": task_type,
            "embedding_strategy": "description + instructions mixed",
            "cocktails": []
        }
        
        for record in result:
            context["cocktails"].append({
                "name": record["name"],
                "description": record["description"],
                "instructions": record["instructions"],
                "recipe_with_measures": record["recipe"],
                "relevance_score": round(record["combined_score"], 4)
            })
        
        return context
    
    def _format_c3_results(self, result, task_type: str) -> Dict:
        """C3 결과 포맷팅"""
        context = {
            "task_type": task_type,
            "embedding_strategy": "description + graph relationships",
            "results": []
        }
        
        for record in result:
            if "family" in task_type.lower():
                context["results"].append({
                    "category_name": record["category_name"],
                    "family_size": record["family_size"],
                    "matches": record["matches"],
                    "avg_similarity": round(record["avg_similarity"], 4)
                })
            else:
                context["results"].append({
                    "name": record["name"],
                    "description": record["description"],
                    "category": record["category"],
                    "ingredients": record["ingredients"],
                    "glass_type": record["glass_type"]
                })
        
        return context
    
    def _format_c4_results(self, result, task_type: str) -> Dict:
        """C4 결과 포맷팅"""
        context = {
            "task_type": task_type,
            "embedding_strategy": "instructions + measure attributes",
            "results": []
        }
        
        for record in result:
            if "recipe" in task_type.lower():
                context["results"].append({
                    "name": record["name"],
                    "instructions": record["instructions"],
                    "detailed_recipe_with_measures": record["detailed_recipe"],
                    "glass_type": record["glass_type"]
                })
            else:
                context["results"].append({
                    "ingredient": record["ingredient_name"],
                    "usage_frequency": record["usage_frequency"],
                    "similarity": round(record["avg_similarity"], 4),
                    "global_popularity": record["global_popularity"]
                })
        
        return context
    
    def _format_context_for_llm(self, context: Dict) -> str:
        """LLM용 컨텍스트를 자연어로 포맷팅"""
        formatted = f"태스크 유형: {context['task_type']}\n"
        formatted += f"검색 전략: {context['embedding_strategy']}\n\n"
        
        if 'cocktails' in context:
            formatted += "검색된 칵테일들:\n"
            for i, cocktail in enumerate(context['cocktails'], 1):
                formatted += f"\n{i}. {cocktail['name']}\n"
                if 'description' in cocktail:
                    formatted += f"   설명: {cocktail['description']}\n"
                if 'ingredients' in cocktail:
                    formatted += f"   재료: {', '.join(cocktail['ingredients'])}\n"
                if 'recipe_with_measures' in cocktail:
                    recipe_items = [f"{item['ingredient']} ({item['measure']})" for item in cocktail['recipe_with_measures']]
                    formatted += f"   레시피: {', '.join(recipe_items)}\n"
                if 'category' in cocktail:
                    formatted += f"   카테고리: {cocktail['category']}\n"
                if 'glass_type' in cocktail:
                    formatted += f"   글라스: {cocktail['glass_type']}\n"
        
        if 'results' in context:
            formatted += "검색 결과:\n"
            for i, result in enumerate(context['results'], 1):
                formatted += f"\n{i}. "
                if 'ingredient' in result:
                    formatted += f"재료: {result['ingredient']} (사용빈도: {result['usage_frequency']})\n"
                elif 'category_name' in result:
                    formatted += f"카테고리: {result['category_name']} (패밀리 크기: {result['family_size']})\n"
                elif 'name' in result:
                    formatted += f"{result['name']}\n"
                    if 'detailed_recipe_with_measures' in result:
                        recipe_items = [f"{item['ingredient']} ({item['measure']})" for item in result['detailed_recipe_with_measures']]
                        formatted += f"   레시피: {', '.join(recipe_items)}\n"
        
        return formatted

    def query_llm(self, context: Dict, user_question: str, task_category: str) -> str:
        """LLM에 컨텍스트 전달하여 답변 생성"""
        formatted_context = self._format_context_for_llm(context)
        
        # 태스크별 전문 프롬프트와 설정 로딩
        system_prompt = self.prompt_loader.get_system_prompt(task_category)
        llm_config = self.prompt_loader.get_llm_config(task_category)
        
        user_message = f"검색 결과:\n{formatted_context}\n\n사용자 질문: {user_question}"
        
        # LLM에 전달되는 실제 컨텍스트 출력
        print(f"\n### LLM에 전달되는 실제 메시지 ###")
        print(f"**System Prompt:**")
        print(system_prompt)
        print(f"\n**User Message:**")
        print(user_message)
        print(f"\n{'='*60}")
        
        try:
            response = openai.chat.completions.create(
                model=llm_config.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 1000)
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM 응답 생성 중 오류 발생: {e}"
    
    def process_query(self, user_question: str, task_category: str, task_type: str):
        """전체 처리 파이프라인"""
        print(f"\n{'='*80}")
        print(f"사용자 질문: {user_question}")
        print(f"태스크 카테고리: {task_category}")
        print(f"태스크 타입: {task_type}")
        print(f"{'='*80}")
        
        # 태스크별 retrieval 실행
        try:
            if task_category == "C1":
                context = self.c1_visual_similarity(user_question, task_type)
            elif task_category == "C2":
                context = self.c2_taste_profile(user_question, task_type)
            elif task_category == "C3":
                context = self.c3_classification(user_question, task_type)
            elif task_category == "C4":
                context = self.c4_recipe_ingredients(user_question, task_type)
            else:
                raise ValueError(f"알 수 없는 태스크 카테고리: {task_category}")
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return None, None
        
        # LLM 답변 생성
        llm_response = self.query_llm(context, user_question, task_category)
        
        # LLM 답변 출력
        print(f"\n### LLM 답변 ###")
        print(llm_response)
        
        return context, llm_response
    
    def close(self):
        """리소스 정리"""
        self.driver.close()

# 사용 예시 및 테스트
if __name__ == "__main__":
    retrieval = CocktailRetrieval()
    
    try:
        # 예시 쿼리들
        test_queries = [
            ("빨간색이고 체리가 올라간 칵테일과 비슷한 칵테일 3가지 추천해줘", "C1", "similar_recommendation"),
            ("네그로니의 맛 프로파일을 달다/시다/쓰다/짜다/감칠맛 0-5 스케일로 알려줘", "C2", "taste_profile"),
            ("다이키리의 분류 정보를 알려줘 - 패밀리, 베이스 스피릿, 스타일, 글라스", "C3", "classification"),
            ("맨해튼 레시피를 정확한 비율과 측정값으로 알려줘", "C4", "recipe_with_measures"),
            ("ABC에 들어갈 핵심 재료들을 추정해줘", "C4", "ingredient_estimation")
        ]
        
        for question, category, task_type in test_queries:
            retrieval.process_query(question, category, task_type)
            print("\n" + "="*100 + "\n")
            
    except Exception as e:
        print(f"전체 실행 중 오류: {e}")
    finally:
        retrieval.close()