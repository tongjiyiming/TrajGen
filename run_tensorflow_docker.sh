sudo docker run -itd --gpus all --name trajgen \
-e NB_USER=lem -e GRANT_SUDO=yes --user root \
-u $(id -u):$(id -g) \
-v $PWD:$HOME/work -w $HOME/work  -p 8893:8888 tongjiyiming/trajgen:igmm
