# Use AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Install ffmpeg and other system dependencies
# AWS Lambda 2023 runtime uses dnf instead of yum
RUN dnf install -y wget tar xz && \
    cd /tmp && \
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg && \
    rm -rf /tmp/ffmpeg* && \
    dnf clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application code
COPY lambdaSpeechToScore.py ${LAMBDA_TASK_ROOT}/
COPY pronunciationTrainer.py ${LAMBDA_TASK_ROOT}/
COPY models.py ${LAMBDA_TASK_ROOT}/
COPY whisper_api_wrapper.py ${LAMBDA_TASK_ROOT}/
COPY ModelInterfaces.py ${LAMBDA_TASK_ROOT}/
COPY RuleBasedModels.py ${LAMBDA_TASK_ROOT}/
COPY WordMatching.py ${LAMBDA_TASK_ROOT}/
COPY WordMetrics.py ${LAMBDA_TASK_ROOT}/
COPY databases/ ${LAMBDA_TASK_ROOT}/databases/

# Set the handler
CMD ["lambdaSpeechToScore.lambda_handler"]

