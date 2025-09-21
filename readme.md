# 칵테일 graph rag 추천 검색 방법

## 🎨 C1: 색상-재료 기반 시각 검색

### 📊 유사도 비교 방법
- **Primary**: `imageDescription_embedding` 코사인 유사도
- **Secondary**: 색상 키워드 ↔ 재료명 임베딩 유사도
- **Final**: 전체 후보 칵테일들의 imageDescription 유사도 랭킹

### 🔍 초기 검색 방법
```python
# Vector similarity search using question embedding
query = """
CALL db.index.vector.queryNodes('cocktail_imageDescription_embedding', $k, $embedding)
YIELD node, score
WHERE score >= $threshold
RETURN node.name as name, score
ORDER BY score DESC
"""
```
- 사용자 질문 전체를 임베딩
- imageDescription_embedding과 벡터 유사도 검색
- 임계값(0.7) 이상인 상위 3개 선정

### 🔄 Hop 수 및 확장 과정
**2-hop 확장:**
1. **1-hop**: 질문 → imageDescription 유사도 → 초기 칵테일 3개
2. **2-hop**: 색상 키워드 → 최고 유사 재료 1개 → 해당 재료를 가진 칵테일 2-3개

### 🎯 최종 선정 방법
1. 초기 후보 + 확장 후보 결합 (중복 제거)
2. 모든 후보의 imageDescription_embedding과 질문 유사도 계산
3. 유사도 기준 내림차순 정렬
4. 상위 3개 최종 선정

---

## 🍷 C2: Glass Type + 계층적 재료 매칭

### 📊 유사도 비교 방법
- **Glass Type**: 임베딩 유사도로 최적 글라스 선택
- **Ingredient**: 키워드별 임베딩 유사도로 실제 재료 매칭
- **Final**: imageDescription_embedding 유사도 기반 레벨별 정렬

### 🔍 초기 검색 방법
```python
# 1. Glass Type 결정
similar_glass_types = self.find_similar_glasstypes_by_name_embedding(glass_keyword, session)

# 2. 해당 Glass Type 칵테일 필터링
MATCH (c:Cocktail)-[:HAS_GLASSTYPE]->(g:GlassType {name: $glass_name})
RETURN c.name as name
```
- 명시된 글라스 타입 우선, 없으면 임베딩 유사도로 추론
- 해당 글라스 타입 칵테일들만 필터링 (100개 정도)

### 🔄 Hop 수 및 확장 과정
**계층적 매칭 (Multi-level):**
```
Level 0: [mint, lime] → 모든 재료를 가진 칵테일
Level 1: [mint] → 1개 재료 제거하고 확장
Level 2: [] → 모든 재료 제거 (글라스 타입만)
```
- 재료를 하나씩 제거하며 점진적 확장
- 목표 후보 수(5개) 달성까지 계속

### 🎯 최종 선정 방법
1. 레벨별로 imageDescription 유사도 계산
2. Level 0 → Level 1 → Level 2 순서로 우선순위
3. 같은 레벨 내에서는 유사도 기준 정렬
4. 상위 3개 최종 선정

---

## 🔗 C3: Multi-hop 재료 확장 검색

### 📊 유사도 비교 방법
- **Name**: 칵테일명 임베딩 유사도 (이름 검색시)
- **Ingredient**: 재료 정확 매치 + 사용 빈도
- **Final**: imageDescription_embedding 유사도 랭킹

### 🔍 초기 검색 방법
```python
# 칵테일 이름 기반
query = """
MATCH (c:Cocktail)
WHERE toLower(c.name) CONTAINS toLower($name)
RETURN c.name as name
LIMIT 3
"""

# 재료 기반
query = """
MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient)
WHERE i.name IN $ingredients
WITH c, count(i) as matched_ingredients
RETURN c.name as name, matched_ingredients
ORDER BY matched_ingredients DESC
"""
```

### 🔄 Hop 수 및 확장 과정
**3-hop 확장:**
```python
query = """
// 1-hop: 사용자 재료들을 가진 칵테일들 찾기
MATCH (c1:Cocktail)-[:HAS_INGREDIENT]->(i1:Ingredient)
WHERE i1.name IN $ingredients

// 2-hop: 그 칵테일들이 공통으로 사용하는 다른 재료들 발견
MATCH (c1)-[:HAS_INGREDIENT]->(i2:Ingredient)
WHERE NOT i2.name IN $ingredients
WITH i2, count(DISTINCT c1) as cocktail_usage_count
WHERE cocktail_usage_count >= $min_usage

// 3-hop: 그 재료들을 사용하는 새로운 칵테일들 탐색
MATCH (c2:Cocktail)-[:HAS_INGREDIENT]->(i2)
"""
```

### 🎯 최종 선정 방법
1. 초기 검색 + Multi-hop 확장 결과 결합
2. 중복 제거 (순서 보존)
3. 모든 후보의 imageDescription과 질문 유사도 계산
4. 유사도 기준 내림차순 정렬 후 상위 4개 선정

---

## 🎯 C4: 칵테일 유사도 및 레시피 대안 추천

### 📊 유사도 비교 방법
- **Target Finding**: 이름 임베딩 유사도 또는 재료 매치율
- **Relationship**: 공유 재료 개수 (그래프 관계 기반)
- **Complexity**: 재료 개수 차이로 복잡도 필터링

### 🔍 초기 검색 방법
```python
# 타겟 칵테일 이름으로 검색
query = """
MATCH (c:Cocktail)
WHERE toLower(c.name) CONTAINS toLower($name)
RETURN c.name as name
LIMIT 1
"""

# 재료 조합으로 타겟 찾기 (폴백)
query = """
MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient)
WHERE i.name IN $ingredients
WITH c, count(i) as matched_count, 
     size((c)-[:HAS_INGREDIENT]->()) as total_ingredients
WHERE matched_count >= 2
RETURN c.name as name, matched_count, total_ingredients,
       toFloat(matched_count) / total_ingredients as match_ratio
ORDER BY match_ratio DESC, matched_count DESC
LIMIT 1
"""
```

### 🔄 Hop 수 및 확장 과정
**1-hop 관계 기반:**
```python
query = """
// 타겟 칵테일의 재료들 찾기
MATCH (c1:Cocktail {name: $target})-[:HAS_INGREDIENT]->(i:Ingredient)
WITH c1, collect(i) as target_ingredients, count(i) as target_count

// 다른 칵테일들과 공유하는 재료 관계 계산
UNWIND target_ingredients as ingredient
MATCH (c2:Cocktail)-[:HAS_INGREDIENT]->(ingredient)
WHERE c2 <> c1
WITH c1, c2, target_count, count(ingredient) as shared_relationships

// 복잡도 필터링
MATCH (c2)-[:HAS_INGREDIENT]->(i2:Ingredient)
WITH c1, c2, target_count, shared_relationships, count(i2) as c2_ingredient_count
WHERE abs(c2_ingredient_count - target_count) <= $complexity_tolerance
AND shared_relationships >= $min_shared
"""
```
- 단일 hop이지만 그래프 관계를 활용한 정교한 계산

### 🎯 최종 선정 방법
1. 공유 재료 개수 기준 내림차순 정렬
2. 같은 공유 재료 개수 내에서는 재료 개수 적은 순 (단순함 우선)
3. 복잡도 허용 범위(±2) 내 칵테일만 선택
4. 상위 3개 최종 선정 + 타겟 칵테일 포함

---

## 📋 알고리즘 비교 요약

| 알고리즘 | 초기 검색 | Hop 수 | 핵심 유사도 | 최종 선정 |
|---------|----------|--------|------------|----------|
| **C1** | imageDescription 벡터 검색 | 2-hop | 시각적 유사도 | imageDescription 랭킹 |
| **C2** | Glass Type 필터링 | Multi-level | 계층적 재료 매칭 | 레벨별 우선순위 |
| **C3** | 이름/재료 직접 매치 | 3-hop | 재료 네트워크 확장 | imageDescription 랭킹 |
| **C4** | 타겟 칵테일 중심 | 1-hop | 공유 재료 개수 | 관계 강도 + 복잡도 |

## 🔧 핵심 특징

- **C1**: 시각적 특성 중심, 색상-재료 매핑 활용
- **C2**: 글라스 타입 제약, 점진적 재료 완화
- **C3**: 재료 네트워크 탐색, 가장 복잡한 확장 로직
- **C4**: 타겟 중심 관계 분석, 복잡도 고려한 대안 추천