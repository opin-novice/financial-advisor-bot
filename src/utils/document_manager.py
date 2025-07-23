import json
import os
import hashlib
from datetime import datetime

class DocumentManager:
    def __init__(self, registry_path='data/document_registry.json'):
        self.registry_path = registry_path
        self.documents = self.load_registry()

    def load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                try:
                    return json.load(f).get('documents', [])
                except json.JSONDecodeError:
                    return []
        return []

    def save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump({'documents': self.documents}, f, indent=4)

    def get_document_by_path(self, file_path):
        for doc in self.documents:
            if doc['file_path'] == file_path:
                return doc
        return None

    def add_or_update_document(self, file_path, category, source='local'):
        file_hash = self.generate_pdf_hash(file_path)
        existing_doc = self.get_document_by_path(file_path)

        if existing_doc:
            if existing_doc.get('hash') != file_hash:
                existing_doc['hash'] = file_hash
                existing_doc['last_modified'] = datetime.now().isoformat()
                existing_doc['category'] = category
                print(f"Document updated: {file_path}")
            else:
                return None # Indicate no update was needed
        else:
            doc = {
                'document_id': len(self.documents) + 1,
                'file_path': file_path,
                'category': category,
                'source': source,
                'hash': file_hash,
                'date_added': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'status': 'processed'
            }
            self.documents.append(doc)
            print(f"Document added: {file_path}")
        
        self.save_registry()
        return self.get_document_by_path(file_path)

    def is_document_changed(self, file_path):
        existing_doc = self.get_document_by_path(file_path)
        if not existing_doc:
            return True  # New document
        
        new_hash = self.generate_pdf_hash(file_path)
        return existing_doc.get('hash') != new_hash

    @staticmethod
    def generate_pdf_hash(file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()