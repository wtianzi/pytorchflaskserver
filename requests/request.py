import requests
#http://192.168.1.121:5000/predict
test_url = 'http://172.28.144.160:5000/predict'

# single file post
resp = requests.post(test_url,files={"file": open('0.bmp','rb')})

# multiple files post
#resp = requests.post(test_url,files={"file1": open('kitten.jpg','rb'),"file2": open('kitten1.jpg','rb')})
print(resp.json())
