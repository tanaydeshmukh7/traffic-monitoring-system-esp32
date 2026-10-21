import urllib.request

url = "http://192.168.4.1:81/stream"

try:
    response = urllib.request.urlopen(url, timeout=5)
    print("Status:", response.status)
    print("Content-Type:", response.headers.get("Content-Type"))
    print("Stream connection successful!")

    data = response.read(1024)
    print("Received bytes:", len(data))

    response.close()

except Exception as e:
    print("Stream connection failed:", repr(e))