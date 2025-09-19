import pandas as pd
from neo4j import GraphDatabase
import openai
import ast
import os
import json
from typing import List, Dict
import time
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

class CocktailGraphBuilder:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the CocktailGraphBuilder with configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Get all sensitive data from environment variables
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USER')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not neo4j_uri:
            raise ValueError("NEO4J_URI not found in environment variables")
        if not neo4j_user:
            raise ValueError("NEO4J_USER not found in environment variables")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD not found in environment variables")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
        openai.api_key = openai_api_key
        self.embedding_model = self.config['embedding_model']
        self.embedding_cache_file = self.config['embedding_cache_file']
        self.embedding_cache = self._load_embedding_cache()
        
    def close(self):
        """Close the Neo4j driver connection and save cache"""
        self._save_embedding_cache()
        self.driver.close()
        
    def _load_embedding_cache(self) -> Dict:
        """Load embedding cache from file"""
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to file"""
        try:
            with open(self.embedding_cache_file, 'w') as f:
                json.dump(self.embedding_cache, f, indent=2)
            print(f"Saved embedding cache to {self.embedding_cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI model with caching
        """
        if not text or pd.isna(text):
            return [0.0] * 1536  # Return zero vector for empty text
        
        # Check cache first
        cache_key = f"{self.embedding_model}:{text}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        try:
            response = openai.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            # Store in cache
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding for text: {e}")
            return [0.0] * 1536
            
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch with caching
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or pd.isna(text):
                embeddings.append([0.0] * 1536)
            else:
                cache_key = f"{self.embedding_model}:{text}"
                if cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        
        # Get embeddings for texts not in cache
        if texts_to_embed:
            print(f"Generating embeddings for {len(texts_to_embed)} texts (using cache for {len(texts) - len(texts_to_embed)} texts)")
            try:
                batch_size = 100
                # Use tqdm for progress bar
                with tqdm(total=len(texts_to_embed), desc="Generating embeddings") as pbar:
                    for i in range(0, len(texts_to_embed), batch_size):
                        batch_texts = texts_to_embed[i:i+batch_size]
                        batch_indices = indices_to_embed[i:i+batch_size]
                        
                        response = openai.embeddings.create(
                            input=batch_texts,
                            model=self.embedding_model
                        )
                        
                        for j, (text, idx) in enumerate(zip(batch_texts, batch_indices)):
                            embedding = response.data[j].embedding
                            embeddings[idx] = embedding
                            # Store in cache
                            cache_key = f"{self.embedding_model}:{text}"
                            self.embedding_cache[cache_key] = embedding
                        
                        pbar.update(len(batch_texts))
                        
                        # Rate limiting
                        time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error getting batch embeddings: {e}")
                # Fill failed embeddings with zero vectors
                for idx in indices_to_embed:
                    if embeddings[idx] is None:
                        embeddings[idx] = [0.0] * 1536
        else:
            print(f"All {len(texts)} embeddings loaded from cache!")
            
        return embeddings
        
    def create_constraints(self):
        """Create unique constraints for the graph"""
        constraints = [
            "CREATE CONSTRAINT cocktail_id IF NOT EXISTS FOR (c:Cocktail) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT glass_name IF NOT EXISTS FOR (g:GlassType) REQUIRE g.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created constraint: {constraint}")
                except Exception as e:
                    print(f"Constraint might already exist: {e}")
                    
    def create_vector_indices(self):
        """Create vector indices for embeddings"""
        indices = [
            """
            CREATE VECTOR INDEX cocktail_description_embedding IF NOT EXISTS
            FOR (c:Cocktail) ON (c.description_embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX cocktail_instructions_embedding IF NOT EXISTS
            FOR (c:Cocktail) ON (c.instructions_embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX cocktail_imageDescription_embedding IF NOT EXISTS
            FOR (c:Cocktail) ON (c.imageDescription_embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """
        ]
        
        with self.driver.session() as session:
            for index in indices:
                try:
                    session.run(index)
                    print(f"Created vector index")
                except Exception as e:
                    print(f"Vector index might already exist: {e}")
                    
    def preprocess_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the cocktail data"""
        # Try utf-8 first, then latin-1 if it fails
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        # 전체 데이터 사용
        print(f"Processing all {len(df)} rows...")
        
        # Fill NaN values with empty strings
        df = df.fillna('')
        
        # Parse ingredients and measures
        def safe_parse(value):
            if not value:
                return []
            try:
                return ast.literal_eval(value)
            except:
                return []
                
        df['ingredients_parsed'] = df['ingredients'].apply(safe_parse)
        df['measures_parsed'] = df['ingredientMeasures'].apply(safe_parse)
        
        return df
        
    def import_cocktails(self, df: pd.DataFrame):
        """Import cocktail nodes with embeddings"""
        print("Generating embeddings for descriptions...")
        descriptions = df['desciription'].tolist()  # Note: typo in original CSV
        description_embeddings = self.get_embeddings_batch(descriptions)
        
        print("Generating embeddings for instructions...")
        instructions = df['instructions'].tolist()
        instruction_embeddings = self.get_embeddings_batch(instructions)
        
        print("Generating embeddings for imageDescription...")
        image_descriptions = df['imageDescription'].tolist() if 'imageDescription' in df.columns else [''] * len(df)
        image_description_embeddings = self.get_embeddings_batch(image_descriptions)
        
        with self.driver.session() as session:
            # Use tqdm for progress bar
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Importing cocktails"):
                # Prepare cocktail data
                cocktail_data = {
                    'id': int(row['id']),
                    'name': row['name'],
                    'alcoholic': row['alcoholic'],
                    'ingredients': row['ingredients'],
                    'drinkThumbnail': row['drinkThumbnail'],
                    'ingredientMeasures': row['ingredientMeasures'],
                    'description': row['desciription'],
                    'description_embedding': description_embeddings[idx],
                    'instructions': row['instructions'],
                    'instructions_embedding': instruction_embeddings[idx],
                    'imageDescription': row.get('imageDescription', ''),
                    'imageDescription_embedding': image_description_embeddings[idx]
                }
                
                # Create cocktail node
                query = """
                MERGE (c:Cocktail {id: $id})
                SET c.name = $name,
                    c.alcoholic = $alcoholic,
                    c.ingredients = $ingredients,
                    c.drinkThumbnail = $drinkThumbnail,
                    c.ingredientMeasures = $ingredientMeasures,
                    c.description = $description,
                    c.description_embedding = $description_embedding,
                    c.instructions = $instructions,
                    c.instructions_embedding = $instructions_embedding,
                    c.imageDescription = $imageDescription,
                    c.imageDescription_embedding = $imageDescription_embedding
                """
                
                session.run(query, cocktail_data)
                    
        print(f"\nImported {len(df)} cocktails successfully!")
        
    def import_ingredients_and_relationships(self, df: pd.DataFrame):
        """Import ingredients, categories, glass types and create relationships"""
        with self.driver.session() as session:
            # Use tqdm for progress bar
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating relationships"):
                cocktail_id = int(row['id'])
                
                # Create Category node and relationship
                if row['category']:
                    session.run("""
                        MERGE (cat:Category {name: $category_name})
                        WITH cat
                        MATCH (c:Cocktail {id: $cocktail_id})
                        MERGE (c)-[:CATEGORY]->(cat)
                    """, {'cocktail_id': cocktail_id, 'category_name': row['category']})
                
                # Create GlassType node and relationship
                if row['glassType']:
                    session.run("""
                        MERGE (g:GlassType {name: $glass_name})
                        WITH g
                        MATCH (c:Cocktail {id: $cocktail_id})
                        MERGE (c)-[:HAS_GLASSTYPE]->(g)
                    """, {'cocktail_id': cocktail_id, 'glass_name': row['glassType']})
                
                # Create Ingredient nodes and relationships
                ingredients = row['ingredients_parsed']
                measures = row['measures_parsed']
                
                for i, ingredient in enumerate(ingredients):
                    if ingredient:
                        measure = measures[i] if i < len(measures) else 'unknown'
                        
                        session.run("""
                            MERGE (ing:Ingredient {name: $ingredient_name})
                            WITH ing
                            MATCH (c:Cocktail {id: $cocktail_id})
                            MERGE (c)-[:HAS_INGREDIENT {measure: $measure}]->(ing)
                        """, {
                            'cocktail_id': cocktail_id,
                            'ingredient_name': ingredient.lower().strip(),  # Normalize ingredient names
                            'measure': measure
                        })
                        
        print(f"\nCreated all relationships successfully!")
        
    def build_graph(self, csv_path: str):
        """Main method to build the entire graph"""
        print("Starting cocktail graph construction...")
        
        # Create constraints and indices
        print("\n1. Creating constraints...")
        self.create_constraints()
        
        print("\n2. Creating vector indices...")
        self.create_vector_indices()
        
        # Preprocess data
        print("\n3. Preprocessing data...")
        df = self.preprocess_data(csv_path)
        print(f"Loaded {len(df)} cocktails")
        
        # Import cocktails
        print("\n4. Importing cocktails with embeddings...")
        self.import_cocktails(df)
        
        # Import ingredients and relationships
        print("\n5. Creating relationships...")
        self.import_ingredients_and_relationships(df)
        
        print("\nGraph construction completed!")
        
    def verify_graph(self):
        """Verify the graph structure with sample queries"""
        with self.driver.session() as session:
            # Count nodes - separate queries to avoid UNION column name issues
            print("\nNode counts:")
            
            result = session.run("MATCH (c:Cocktail) RETURN count(c) as count")
            print(f"  Cocktails: {result.single()['count']}")
            
            result = session.run("MATCH (i:Ingredient) RETURN count(i) as count")
            print(f"  Ingredients: {result.single()['count']}")
            
            result = session.run("MATCH (cat:Category) RETURN count(cat) as count")
            print(f"  Categories: {result.single()['count']}")
            
            result = session.run("MATCH (g:GlassType) RETURN count(g) as count")
            print(f"  Glass Types: {result.single()['count']}")
            
                
            # Count relationships - separate queries
            print("\nRelationship counts:")
            
            result = session.run("MATCH ()-[r:HAS_INGREDIENT]->() RETURN count(r) as count")
            print(f"  HAS_INGREDIENT: {result.single()['count']}")
            
            result = session.run("MATCH ()-[r:CATEGORY]->() RETURN count(r) as count")
            print(f"  CATEGORY: {result.single()['count']}")
            
            result = session.run("MATCH ()-[r:HAS_GLASSTYPE]->() RETURN count(r) as count")
            print(f"  HAS_GLASSTYPE: {result.single()['count']}")
                
            # Sample cocktail with all relationships
            results = session.run("""
                MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient)
                WHERE c.name = '151 Florida Bushwacker'
                RETURN c.name as cocktail, collect(i.name) as ingredients
                LIMIT 1
            """)
            
            print("\nSample cocktail:")
            for record in results:
                print(f"  Cocktail: {record['cocktail']}")
                print(f"  Ingredients: {record['ingredients']}")


if __name__ == "__main__":
    # Create graph builder with config file
    builder = CocktailGraphBuilder("config.json")
    
    try:
        # Build the graph
        builder.build_graph("cocktail_data_436_final.csv")
        
        # Verify the graph
        builder.verify_graph()
        
    finally:
        builder.close()