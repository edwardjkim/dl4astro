Run docker with:

docker run -d --name ipython -p 443:8888 -e "PASSWORD=YourPassword" -v ~/shared:/notebooks/shared edwardjkim/scipyserver
