# urllib, requests를 이용한 crawling
# @date 2019.11.01
# @author k2moon72@gmail.com
# @param url, params, headers, method, encoding, return


try:
    from urllib.request import Request, urlopen
    from urllib import parse
    from datetime import datetime
    import time
    import requests
    
except Exception as e:
    str_e = str(e)
    print(e)
    print('pip install ' + str_e.split('\'')[1])

def urllib_urlopen(
        url='',
        params='',
        headers={},
        method='GET',
        encoding='utf-8',
        api = False
        ):
    try:
        if not api:
            time.sleep(1)
            
        method.upper()
        if method == 'POST':
            params = parse.urlencode(params).encode('utf-8')
            request = Request(url, headers=headers, data=params, method='POST')
        else:
            params = parse.urlencode(params)
            url = url + '?' + params
            request = Request(url, headers=headers)
        
        response = urlopen(request)
        
        try:
            response_read = response.read()
            ret_data = response_read.decode(encoding)
        except UnicodeDecodeError:
            ret_data = response_read.decode(encoding, 'replace')

        print('{} : success for request {}'.format(datetime.now(), url))
        return ret_data
    except Exception as e:
        print(e) # HTTP Error 404: Not Found
        
def requests_get(
        url='',
        params='',
        headers='',
        method='GET',
        encoding='utf-8',
        _type = 'text',
        ssl = False,
        api = False
        ):
    try:
        if not api:
            time.sleep(1)
            
        method.upper()
        
        verify = True
        
        if ssl:
            verify = False
            
        if method == 'POST':
             # verify=False : SSL 확인 않함
            response = requests.get(url, headers=headers, data=params, verify=verify)
        else:
            response = requests.get(url, headers=headers, params=params, verify=verify)
        
        try:
            response.encoding = encoding
            if _type == 'json':
                ret_data = response.json()
            else:
                ret_data = response.text
        except UnicodeDecodeError:
            ret_data = response.text  
        print('{} : success for request {}'.format(datetime.now(), url))
        return ret_data
    except Exception as e:
        print(e) # HTTP Error 404: Not Found