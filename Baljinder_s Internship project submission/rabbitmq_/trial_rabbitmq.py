# Import necessary libraries
import pika
import json

# RabbitMQ connection details
RABBITMQ_HOST = '192.168.140.155'  # Replace with your RabbitMQ hostname or IP address
RABBITMQ_PORT = 5672         # Default RabbitMQ port
USERNAME = 'admin'           # Replace with your RabbitMQ username
PASSWORD = 'admin'           # Replace with your RabbitMQ password
QUEUE_NAME = 'Prediction_Analysis'
EXCHANGE = ''
ROUTING_KEY = 'Prediction_Analysis'

# Function to publish a dummy message to RabbitMQ
def publish_dummy_message():
    # Set up credentials for the connection
    credentials = pika.PlainCredentials(USERNAME, PASSWORD)

    # Establish a connection to RabbitMQ with the given credentials
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        )
    )
    channel = connection.channel()

    # Declare the queue to ensure it exists
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    # Dummy message to publish
    dummy_message = {
        "message": "Hello, this is a test message with authentication!"
    }
    
    # Convert the message to JSON format
    json_message = json.dumps(dummy_message)

    # Publish the message to the queue
    channel.basic_publish(
        exchange=EXCHANGE,
        routing_key=ROUTING_KEY,
        body=json_message,
        properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
    )

    print(f"Message published to queue '{QUEUE_NAME}': {json_message}")

    # Close the connection
    connection.close()

# Call the function to publish the dummy message
publish_dummy_message()
