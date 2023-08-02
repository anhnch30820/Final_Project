## Docker build:
```bash
dokcer build . -t $name_image
```

## Docker run:
```bash
docker run -it -p 8501:8501 $name_image
```

### Image đã build trên Docker hub
```bash
docker run -it -p 8501:8501 hoanganh123456/deploy_project:20222
```