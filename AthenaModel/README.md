Too build the docker file run 
sudo docker build --pull --rm -f "Dockerfile" -t videocreator:latest "." 

To then run the docker file on a local machine
sudo docker run --gpus all videocreator