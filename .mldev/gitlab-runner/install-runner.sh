# this uses docker as container engine

docker volume create gitlab-runner
docker run --rm --name gitlab-runner -v gitlab-runner:/etc/gitlab-runner -v /var/run/docker.sock:/var/run/docker.sock -it gitlab/gitlab-runner register

# this starts the runner in interactive mode
docker run --rm --name gitlab-runner -v gitlab-runner:/etc/gitlab-runner -v /var/run/docker.sock:/var/run/docker.sock -it gitlab/gitlab-runner

# this starts the runner as daemon
docker run -d --restart always --name gitlab-runner -v gitlab-runner:/etc/gitlab-runner -v /var/run/docker.sock:/var/run/docker.sock gitlab/gitlab-runner
