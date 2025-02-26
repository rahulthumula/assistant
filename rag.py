from database import CosmosDB
from embeddings import EmbeddingGenerator
from search import VectorStore
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
import uuid

from tenacity import retry, stop_after_attempt, wait_exponential

class RAGAssistant:
    def __init__(self, user_id):
        print(f"\nDebug - Initializing RAGAssistant for user {user_id}")
        self.user_id = user_id
        self.cosmos_db = CosmosDB()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(user_id)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_embedding_with_retry(self, text):
        """Generate embedding with retry logic."""
        try:
            print(f"\nDebug - Generating embedding for text (first 50 chars): {text[:50]}...")
            embedding = await self.embedding_generator.generate_embedding(text)
            print(f"Debug - Generated embedding dimensions: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

    def _verify_dimensions(self, embedding, context=""):
        """Verify embedding dimensions"""
        expected_dim = self.embedding_generator.expected_dim
        actual_dim = len(embedding)
        print(f"Debug - Verifying dimensions {context}:")
        print(f"Debug - Expected: {expected_dim}, Actual: {actual_dim}")
        if actual_dim != expected_dim:
            raise ValueError(f"Dimension mismatch {context}: Expected {expected_dim}, got {actual_dim}")

    def _create_item_content(self, item):
        """Create rich, searchable content for an inventory item."""
        try:
            print(f"\nDebug - Creating content for item: {item.get('Inventory Item Name', 'Unknown')}")
            
            # Primary information in natural language
            primary_desc = f"This is {item.get('Inventory Item Name', 'Unknown Item')}, "
            primary_desc += f"a {item.get('Category', 'unknown').lower()} product "
            if item.get('Brand'):
                primary_desc += f"from {item.get('Brand')}. "
            primary_desc += f"The full product name is {item.get('Item Name', 'Unknown')}. "
            
            # Pricing information
            price_desc = f"It costs ${item.get('Case Price', 0)} per {item.get('Priced By', 'unit').replace('per ', '')}. "
            price_desc += f"Each unit costs ${item.get('Cost of a Unit', 0)}. "
            
            # Quantity information
            quantity_desc = f"Each case contains {item.get('Quantity In a Case', 0)} {item.get('Measured In', 'units')}. "
            quantity_desc += f"Total available units are {item.get('Total Units', 0)}. "
            
            # Additional specifications
            specs = f"The item number is {item.get('Item Number', 'unknown')}. "
            if item.get('Splitable', 'NO') == "NO":
                specs += "This item cannot be split. "
            
            # Category-specific details
            category_desc = ""
            category = item.get('Category', '').upper()
            if category == "DAIRY":
                category_desc = "This is a dairy product that should be stored refrigerated. "
            elif category == "FROZEN":
                category_desc = "This is a frozen product that must be kept frozen. "
            elif category == "PRODUCE":
                category_desc = "This is a fresh produce item. "

            # Combine all content
            content = f"""Product Overview:
{primary_desc}

Pricing Details:
{price_desc}

Quantity Information:
{quantity_desc}

Specifications:
{specs}
{category_desc}"""

            print(f"Debug - Created content successfully")
            return content
        except Exception as e:
            print(f"Error creating content: {str(e)}")
            raise

    async def index_inventory_items(self, inventory_list):
     vector_documents = []
     print(f"Debug - Processing {len(inventory_list)} items")
    
     if not inventory_list or len(inventory_list) == 0:
        raise ValueError("No inventory documents found")
        
     inventory_doc = inventory_list[0]
     items = inventory_doc.get('items', [])
    
     for i, item in enumerate(items):
        try:
            # Create rich content
            content = self._create_item_content(item)
            
            # Generate embedding
            embedding = await self.embedding_generator.generate_embedding(content)
            
            # Create document with correct field mapping
            vector_doc = {
    'id': str(uuid.uuid4()),
    'userId': self.user_id,
    # Use the exact keys from the source data
    'supplier_name': item['Supplier Name'],  # This matches the field in the source
    'inventory_item_name': item['Inventory Item Name'],  # This matches the field in the source
    'item_name': item['Item Name'],
    'item_number': item['Item Number'],
    'quantity_in_case': float(item['Quantity In a Case']),
    'total_units': float(item['Total Units']),
    'case_price': float(item['Case Price']),
    'cost_of_unit': float(item['Cost of a Unit']),
    'category': item['Category'],
    'measured_in': item['Measured In'],
    'catch_weight': item['Catch Weight'],
    'priced_by': item['Priced By'],
    'splitable': item['Splitable'],
    'content': content,
    'content_vector': embedding
}
            vector_documents.append(vector_doc)
            
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            continue
    
     if vector_documents:
        await self.vector_store.add_documents(vector_documents)

    async def initialize(self):
        """Initialize the RAG system."""
        try:
            # Get user inventory
            inventory = await self.cosmos_db.get_user_documents(self.user_id)
            if not inventory:
                raise ValueError(f"No inventory found for user {self.user_id}")
            
            print(f"\nDebug - Creating search index")
            await self.vector_store.create_index()
            
            print(f"\nDebug - Indexing inventory items")
            await self.index_inventory_items(inventory)
            
            print("Debug - Initialization completed successfully")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def _generate_search_terms(self, item):
        """Generate alternative search terms and synonyms for the item."""
        search_terms = []
        
        # Add main terms
        inventory_item_name = item.get('Inventory Item Name', '').lower()
        if inventory_item_name:
            name_parts = inventory_item_name.split(',')
            search_terms.extend(name_parts)
        
        # Add category-specific synonyms
        category = item.get('Category', '').upper()
        if category == "DAIRY":
            search_terms.extend(["dairy product", "dairy item", "refrigerated"])
            if "cheese" in inventory_item_name:
                search_terms.extend(["cheese product", "dairy cheese"])
            if "milk" in inventory_item_name:
                search_terms.extend(["milk product", "dairy milk"])
        
        # Add pricing-related terms
        priced_by = item.get('Priced By', '').lower()
        if "case" in priced_by:
            search_terms.extend(["case pricing", "bulk item"])
        elif "pound" in priced_by:
            search_terms.extend(["pound pricing", "weight based"])
        
        # Add brand-related terms
        brand = item.get('Brand', '')
        if brand:
            search_terms.append(f"{brand} brand")
            search_terms.append(f"{brand} product")
        
        return ", ".join(search_terms)

    async def query(self, user_question, top_k=5):
        question_embedding = await self.embedding_generator.generate_embedding(user_question)
        search_results = await self.vector_store.search(question_embedding, top_k)
        
        prompt = f"""You are an inventory assistant helping with questions about the user's inventory items.
Base your answer only on the provided inventory data.
Inventory Items:
{search_results}
Question: {user_question}
."""
        
        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful inventory assistant that provides accurate information about inventory items, prices, and quantities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=15000
        )
        
        return response.choices[0].message.content