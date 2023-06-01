# Coco-Detect


- A HTML, CSS, JS project.
- Should be served using a web server or serving tool like live server for VS code.
- ```BASE_URL``` should be changed to the specific backend

For flask backend
```js
const BASE_URL = "http://localhost:5000"
```


- ```PREDICT_ENDPOINT``` is the appending endpoint to the base url which should be the api endpoint of the backend


- if the api endpoint is ```/predict```
```js
const PREDICT_ENDPOINT = "/predict"
```

- Backend API should accept an image as FormData with,
    - key,value pair : ```"image", <imageFile>``` 