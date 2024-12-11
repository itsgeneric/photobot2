# Use the Python 3.9 AWS Lambda base image
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies and update CMake
RUN yum update -y && \
    yum install -y cmake3 gcc gcc-c++ make \
    boost-devel libX11 libXext wget && \
    # Remove existing cmake if needed
    yum remove -y cmake && \
    # Create symlink for cmake3
    ln -s /usr/bin/cmake3 /usr/bin/cmake && \
    yum clean all

# Set working directory in the container
WORKDIR /var/task

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install dlib and other build dependencies
RUN pip install --no-cache-dir dlib

# Copy the rest of the application files into the container
COPY . .

# Set the CMD to the Lambda Python runtime
CMD [ "main.handler" ]