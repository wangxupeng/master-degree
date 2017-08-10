import requests
import re
def crawler163():
    content = requests.get('http://www.163.com/').text
    pattern1 = re.compile('<div class="tab_main clearfix".*?</ul>', re.S)
    results_part = re.findall(pattern1, content)
    pattern2 = re.compile('<li.*?href="(.*?)">(.*?)</a>', re.S)
    results_filter = re.findall(pattern2,str(results_part))
    for result in results_filter:
        http,title = result
        http = re.sub('\s', '', http)
        title = re.sub('\s', '', title)
        print(http,title)

if __name__ == "__main__" :
    crawler163()
