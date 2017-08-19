#coding=utf-8
import json
import os
import re
from hashlib import md5
from urllib.parse import urlencode
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import pyodbc

def get_page_index(offset,keyword):

    url = 'http://www.toutiao.com/search_content/?offset={}&format=json&keyword={}&autoload=true&count=20&cur_tab=3'.format(offset,keyword)
    response = requests.get(url)
    try:
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        print('请求索引页出错')
        return None

def parse_page_index(html):
    data = json.loads(html)
    if data and 'data' in data.keys():
        for item in data.get('data'):
            yield item.get('article_url')


def get_page_detail(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        print('请求详情页出错')
        return None

def parse_page_detail(html,url):
    soup = BeautifulSoup(html,'html.parser')
    title = soup.select('title')[0].get_text()
    images_pattern = re.compile('gallery: (.*?),\n',re.S)
    result = re.search(images_pattern,html)
    if result:
        data = json.loads(result.group(1))
        if data and 'sub_images' in data.keys():
            sub_images = data.get('sub_images')
            print(title)
            images = [item.get('url') for item in sub_images]
            for image in images: download_image(image)
            return {
                'title':title,
                'url': url,
                'images':images
            }

def save_to_mytsql(result):
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;DATABASE=test;UID=sa;PWD=123456')
    cur = conn.cursor()
    sql='insert into toutiao([title],[url], [images]) values(?,?,?)'
    cur.execute(sql,(result['title'], result['url'], str(result['images'])))
    conn.commit()
    conn.close()

def download_image(url):
    print('正在下载:',url)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            save_image(response.content)#图片一般使用content
        return None
    except RequestException:
        print('请求图片出错',)
        return None


def save_image(content):
    file_path = '{0}/{1}.{2}'.format(os.getcwd(),md5(content).hexdigest(),'jpg')
    if not os.path.exists(file_path):
        with open(file_path,'wb') as f:
            f.write(content)
            f.close()


def main():
    html = get_page_index(0,"街拍")
    for url in parse_page_index(html):
        html = get_page_detail(url)
        if html:
            result = parse_page_detail(html,url)
            if result != None:
                save_to_mytsql(result)

if __name__ == "__main__":
    main()
