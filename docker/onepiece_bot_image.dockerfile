# Use a base image with Python
FROM python:3.9

# Set a working directory
WORKDIR app/

# Copy your bot code and any other necessary files to the container
COPY . ./

# Install required dependencies
RUN pip install python-telegram-bot chromadb tqdm langchain openai tiktoken

# Set the command to run your bot when the container starts
CMD ["python", "src/telegram_bot.py"]
