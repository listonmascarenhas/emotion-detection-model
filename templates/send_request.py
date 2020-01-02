import requests
url = "http://192.168.0.220:5000/uploader"
path = "C:\\Users\\theco\\Desktop\\New folder\\00.txt"
files = {'file': open(path)}
r = requests.post(url, files=files)
print(r)