# UBA Medicine Chatbot

An educational chatbot for the Faculty of Medicine of the University of Buenos Aires (UBA). This project implements a WhatsApp-based chatbot that uses RAG (Retrieval Augmented Generation) to provide accurate and contextual responses to medical education queries.

## Features

- WhatsApp Business API integration
- RAG-based response generation
- Fine-tuned language model
- Automatic setup and deployment
- Robust error handling
- Detailed logging

## Project Structure

```
.
├── data/
│   ├── raw/           # Raw data files
│   ├── processed/     # Processed data
│   └── embeddings/    # Generated embeddings
├── models/            # Model files
├── scripts/           # Python scripts
│   ├── auto_setup.py  # Automatic setup
│   ├── create_embeddings.py  # Embedding generation
│   ├── deploy_backend.py     # Backend deployment
│   ├── preprocess.py  # Data preprocessing
│   ├── run_rag.py     # RAG system
│   └── train_finetune.py  # Model fine-tuning
└── logs/              # Log files
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chatbot_uba.git
cd chatbot_uba
```

2. Run the automatic setup:
```bash
python scripts/auto_setup.py
```

3. Configure environment variables in `.env`:
```
ENVIRONMENT=development
MODEL_PATH=models/finetuned_model
EMBEDDINGS_DIR=data/embeddings
WHATSAPP_API_TOKEN=your_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_id
WHATSAPP_BUSINESS_ACCOUNT_ID=your_account_id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_verify_token
MY_PHONE_NUMBER=your_test_number
```

## Development

1. Start the backend server:
```bash
python scripts/deploy_backend.py
```

2. Generate embeddings:
```bash
python scripts/create_embeddings.py
```

3. Fine-tune the model:
```bash
python scripts/train_finetune.py
```

## Testing

1. Send a test message:
```bash
curl http://localhost:8000/test-message
```

2. Check service health:
```bash
curl http://localhost:8000/health
```

## Deployment

1. Configure production environment variables
2. Deploy backend:
```bash
python scripts/deploy_backend.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Faculty of Medicine, UBA
- WhatsApp Business API
- Hugging Face Transformers
- FastAPI