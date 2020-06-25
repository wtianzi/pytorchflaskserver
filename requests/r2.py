from __future__ import print_function
import requests
import json
import cv2

#test_url = 'http://192.168.1.121:5000/predict'
test_url = 'http://172.28.144.160:5000/predict'
# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('kitten.jpg')
#res,img_byte=cv2.imencode('jpeg',img)

response = requests.post(test_url, img, headers=headers)
# decode response
print(json.loads(response.text))
