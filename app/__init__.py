import logging

from flask import Flask

app = Flask(__name__)

app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024    # set maximum file size to 16 MiB
)

file_handler = logging.FileHandler('submission.log')
app.logger.addHandler(file_handler)

from app import routes
