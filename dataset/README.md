
docker build -t get_dataset .

docker run -v $(pwd)/bags:/app/bags -it get_dataset bash
