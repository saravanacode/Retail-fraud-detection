from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
client = MongoClient('mongodb+srv://saroo:sarojini@students.jbrazv2.mongodb.net/?retryWrites=true&w=majority&appName=students')
db = client['fraud_detection']
transactions = db['transactions']

# Delete all data from the transactions collection
def clear_transactions():
    result = transactions.delete_many({})
    print(f"Deleted {result.deleted_count} documents from transactions collection")

# Call the function to clear all data
clear_transactions()

@app.route('/save_result', methods=['POST'])
def simulate_transaction():
    data = request.json
    customer_id = data['customer_id']

    
    # Check if customer_id already exists
    existing_transaction = transactions.find_one({'customer_id': customer_id})
    if existing_transaction is None:
        transaction = {
            'customer_id': customer_id,
            'transaction_time': datetime.utcnow(),
            'invoice_generated': data['invoice_provided']
        }
        print(transaction)
        result = transactions.insert_one(transaction)
        return jsonify({'message': 'Transaction simulated', 'id': str(result.inserted_id)}), 200
    else:
        print(f"Transaction for customer_id {customer_id} already exists. Ignoring.")
        result = None
    
        return jsonify({'message': 'Transaction already exists', 'id': str(customer_id)}), 200

if __name__ == '__main__':
    app.run(debug=True)