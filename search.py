# search.py
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField,
)
from config import (
    SEARCH_SERVICE_ENDPOINT,
    SEARCH_SERVICE_KEY,
    OPENAI_EMBEDDING_MODEL
)

class VectorStore:
    def __init__(self, user_id):
        print(f"\nDebug - Initializing VectorStore for user {user_id}")
        self.credential = AzureKeyCredential(SEARCH_SERVICE_KEY)
        self.index_name = f"inventory-{user_id}"
        self.index_client = SearchIndexClient(
            endpoint=SEARCH_SERVICE_ENDPOINT,
            credential=self.credential
        )
        try:
         if self.index_name in list(self.index_client.list_index_names()):
            print(f"Debug - Connecting to existing index: {self.index_name}")
            self.search_client = SearchClient(
                endpoint=SEARCH_SERVICE_ENDPOINT,
                credential=self.credential,
                index_name=self.index_name
            )
         else:
            self.search_client = None
        except Exception as e:
         print(f"Error checking index: {str(e)}")
        self.search_client = None
    

    async def create_index(self):
        try:
            # Try to delete existing index
            try:
                self.index_client.delete_index(self.index_name)
                print(f"Debug - Deleted existing index: {self.index_name}")
            except Exception as e:
                print(f"Debug - No existing index to delete: {str(e)}")

            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-config",
                        kind="hnsw",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="hnsw-config"
                    )
                ]
            )

            # Updated field definitions to match the document structure
            fields = [
                SimpleField(name="id", type="Edm.String", key=True),
                SimpleField(name="userId", type="Edm.String", filterable=True),
                SearchableField(name="supplier_name", type="Edm.String", filterable=True,searchable=True),
                SearchableField(name="inventory_item_name", type="Edm.String", filterable=True, searchable=True),
                SearchableField(name="item_name", type="Edm.String", filterable=True),
                SimpleField(name="item_number", type="Edm.String", filterable=True),
                SimpleField(name="quantity_in_case", type="Edm.Double"),
                SimpleField(name="total_units", type="Edm.Double"),
                SimpleField(name="case_price", type="Edm.Double"),
                SimpleField(name="cost_of_unit", type="Edm.Double"),
                SearchableField(name="category", type="Edm.String", filterable=True),
                SearchableField(name="measured_in", type="Edm.String"),
                SimpleField(name="catch_weight", type="Edm.String"),
                SearchableField(name="priced_by", type="Edm.String", filterable=True),
                SimpleField(name="splitable", type="Edm.String", filterable=True),
                SearchableField(name="content", type="Edm.String"),
                SearchField(
                    name="content_vector",
                    type="Collection(Edm.Single)",
                    vector_search_dimensions=1536,
                    vector_search_profile_name="vector-profile"
                )
            ]

            print(f"Debug - Creating new index with fields: {[f.name for f in fields]}")
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            self.index_client.create_or_update_index(index)
            print(f"Debug - Successfully created index: {self.index_name}")
            
            self.search_client = SearchClient(
                endpoint=SEARCH_SERVICE_ENDPOINT,
                credential=self.credential,
                index_name=self.index_name
            )
            return True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            raise

    async def add_documents(self, documents):
     try:
        print(f"\nDebug - Processing {len(documents)} documents for upload")
        processed_docs = []
        
        for i, doc in enumerate(documents):
            print(f"\nDebug - Processing document {i+1}:")
            print(f"Original document keys: {list(doc.keys())}")
            print(f"\nDebug - Document {i} pre-processing:")
            print(f"Supplier Name (original): {doc.get('Supplier Name', 'NOT_FOUND')}")
            print(f"supplier_name (mapped): {doc.get('supplier_name', 'NOT_FOUND')}")
            print(f"Inventory Item Name (original): {doc.get('Inventory Item Name', 'NOT_FOUND')}")
            print(f"inventory_item_name (mapped): {doc.get('inventory_item_name', 'NOT_FOUND')}")
            
            # Process document with exact field mapping
            processed_doc = {
                'id': doc['id'],
                'userId': doc['userId'],
                'supplier_name': doc.get('supplier_name', doc.get('Supplier Name', '')),  # Try both formats
                'inventory_item_name': doc.get('inventory_item_name', doc.get('Inventory Item Name', '')),
                'item_name': doc['item_name'],
                'item_number': doc['item_number'],
                'quantity_in_case': float(doc.get('quantity_in_case', doc.get('Quantity In a Case', 0))),
                'total_units': float(doc['total_units']),
                'case_price': float(doc['case_price']),
                'cost_of_unit': float(doc.get('cost_of_unit', doc.get('unit_cost', 0))),
                'category': doc['category'],
                'measured_in': doc['measured_in'],
                'catch_weight': doc['catch_weight'],
                'priced_by': doc['priced_by'],
                'splitable': doc['splitable'],
                'content': doc['content'],
                'content_vector': doc['content_vector']
            }
            
            # Debug print processed document
            print("\nDebug - Processed document fields:")
            for key, value in processed_doc.items():
                if key != 'content_vector':  # Skip printing the vector
                    print(f"{key}: {value}")
            
            processed_docs.append(processed_doc)
        
        if processed_docs:
            print(f"\nDebug - Uploading {len(processed_docs)} documents to search index")
            try:
                result = self.search_client.upload_documents(documents=processed_docs)
                print(f"Debug - Upload result: {result}")
                return result
            except Exception as e:
                print(f"Error during upload: {str(e)}")
                print("Debug - First processed document structure:")
                print(processed_docs[0].keys())
                raise
            
     except Exception as e:
        print(f"Error processing documents: {str(e)}")
        print(f"Full error details: {e}")
        raise
    async def search(self, query_vector, top_k=3):
        try:
            # Updated field names in select
            select_fields = ",".join([
                "inventory_item_name",
                "item_name",
                "category",
                "case_price",
                "cost_of_unit",
                "total_units",
                "measured_in",
                "priced_by",
                "content"
            ])
            
            results = self.search_client.search(
                search_text=None,
                 vector_queries=[{
                    'vector': query_vector,
                    'fields': 'content_vector',
                    'k': top_k,
                    'kind': 'vector'
                }],
                select=select_fields
            )
            
            return [dict(result) for result in results]
        except Exception as e:
            print(f"Error performing search: {str(e)}")
            raise
    async def connect_to_index(self):
     try:
        self.search_client = SearchClient(
            endpoint=SEARCH_SERVICE_ENDPOINT,
            credential=self.credential,
            index_name=self.index_name
        )
        print(f"Debug - Connected to existing index: {self.index_name}")
        return True
     except Exception as e:
        print(f"Error connecting to index: {str(e)}")
        raise    