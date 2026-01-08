# LangChain과 Bright Data를 활용한 Webスクレイピング

[![Promo](https://github.com/bright-kr/LinkedIn-Scraper/raw/main/Proxies%20and%20scrapers%20GitHub%20bonus%20banner.png)](https://brightdata.co.kr/) 

이 가이드는 Webスクレイピング과 LangChain을 결합하여 실제 환경의 LLM 데이터 강화(data enrichment)를 수행하는 방법을, 자세한 단계별 안내로 설명합니다.

- [Web Scraping을 사용하여 LLM 애플리케이션을 강화하기](#using-web-scraping-to-power-your-llm-applications)
- [LangChain에서 스크랩된 데이터 사용의 이점과 과제](#benefits-and-challenges-of-using-scraped-data-in-langchain)
- [Bright Data로 구동되는 LangChain Web Scraping: 단계별 가이드](#langchain-web-scraping-powered-by-bright-data-step-by-step-guide)
  - [사전 준비 사항](#prerequisites)
  - [단계 #1: 프로젝트 설정](#step-1-project-setup)
  - [단계 #2: 필요한 라이브러리 설치](#step-2-install-the-required-libraries)
  - [단계 #3: 프로젝트 준비](#step-3-prepare-your-project)
  - [단계 #4: Web Scraper API 구성](#step-4-configure-web-scraper-api)
  - [단계 #5: Web Scraping에 Bright Data 사용](#step-5-use-bright-data-for-web-scraping)
  - [단계 #6: OpenAI 모델 사용 준비](#step-6-get-ready-to-use-openai-models)
  - [단계 #7: LLM 프롬프트 생성](#step-7-generate-the-llm-prompt)
  - [단계 #8: OpenAI 통합](#step-8-integrate-openai)
  - [단계 #9: AI 처리 데이터 내보내기](#step-9-export-the-ai-processed-data)
  - [단계 #10: 로그 추가](#step-10-add-logs)
  - [단계 #11: 전체 구성 결합](#step-11-put-it-all-together)
- [결론](#conclusion)


## Using Web Scraping to Power Your LLM Applications

Webスクレイピング은 웹사이트에서 데이터를 추출하여 RAG([Retrieval-Augmented Generation](https://brightdata.co.kr/blog/web-data/rag-explained)) 애플리케이션을 구동하고, LLM([Large Language Models](https://www.ibm.com/think/topics/large-language-models))을 활용할 수 있게 합니다. 이는 정적 데이터베이스와, 이러한 애플리케이션에 필요한 실시간/도메인 특화/대규모 データセット 간의 격차를 메웁니다.

## Benefits and Challenges of Using Scraped Data in LangChain

[LangChain](https://www.langchain.com/)은 분석, 요약, Q&A와 같은 작업을 위해 LLM을 다양한 데이터 소스와 통합합니다. 그러나 アンチボット 조치, CAPTCHA, 동적 웹사이트로 인해 고품질 데이터를 수집하는 것은 어렵습니다. Bright Data의 [Web Scraper API](https://brightdata.co.kr/products/web-scraper)는 IP 회전, CAPTCHA 해결, JavaScript 렌더링과 같은 기능으로 이러한 문제를 해결하여, 단순한 API 호출만으로 효율적이고 신뢰할 수 있는 데이터 수집을 보장합니다.

## LangChain Web Scraping Powered By Bright Data: Step-by-Step Guide

Bright Data의 Web Scraper API를 사용해 CNN 기사에서 콘텐츠를 가져오는 LangChain Webスクレイピング 스크립트를 구축한 다음, 이를 OpenAI로 보내 요약하는 방법을 알아봅니다. 대상은 [이 CNN 기사](https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/)를 사용합니다.

![CNN article on Christmas](https://github.com/bright-kr/langchain-web-scraping/blob/main/Images/image-131-1024x492.png)

이 간단한 예시는 SERP 데이터 기반 AG 챗봇 생성과 같은 추가 LangChain 기능으로 쉽게 확장할 수 있습니다.

### Prerequisites

이 가이드를 진행하려면 다음이 필요합니다:

- 머신에 설치된 Python 3+
- OpenAI API key
- Bright Data 계정

### Step #1: Project Setup

Python 3이 설치되어 있는지 확인합니다. 그런 다음 프로젝트용 폴더를 생성합니다:

```bash
mkdir langchain_scraping
```

`langchain_scrping`에는 Python LangChain スクレイピング 프로젝트가 포함됩니다.

그다음 프로젝트 폴더로 이동하여 그 안에 Python 가상 환경을 초기화합니다:

```bash
cd langchain_scraping
python3 -m venv env
```

> **Note**:
>
> Windows에서는 `python3` 대신 `python`을 사용합니다.

이제 선호하는 Python IDE에서 프로젝트 디렉터리를 열고, `langchain_scraping` 안에 `script.py` 파일을 추가합니다.

가상 환경을 활성화합니다:

```bash
./env/bin/activate
```

또는 Windows에서는 다음을 실행합니다:

```bash
env/Scripts/activate
```

### Step #2: Install the Required Libraries

Python LangChain スクレイピング 프로젝트는 다음 라이브러리에 의존합니다:

- [`python-dotenv`](https://pypi.org/project/python-dotenv/): `.env` 파일에서 환경 변수를 로드하기 위함입니다. Bright Data 및 OpenAI 자격 증명과 같은 민감한 정보를 관리하는 데 사용됩니다.
- [`requests`](https://pypi.org/project/requests/): Bright Data의 Web Scraper API와 상호작용하기 위한 HTTP リクエスト 수행에 사용됩니다.
- [`langchain_openai`](https://pypi.org/project/langchain-openai/): [`openai`](https://pypi.org/project/openai/) SDK를 통한 OpenAI용 LangChain 통합입니다.

활성화된 가상 환경에서 모든 의존성을 설치합니다:

```bash
pip install python-dotenv requests langchain-community
```

### Step #3: Prepare Your Project

`scripts.py`에 다음 import를 추가합니다:

```python
from dotenv import load_dotenv
import os
```

이 두 줄을 통해 환경 변수 파일을 읽을 수 있습니다.

프로젝트 폴더에 `.env` 파일을 생성하여 모든 자격 증명을 저장합니다.

`script.py`에서 `python-dotenv`가 `.env`의 환경 변수를 로드하도록 지시합니다:

```python
load_dotenv()
```

이제 다음과 같이 `.env` 파일 또는 시스템에서 환경 변수를 읽을 수 있습니다:

```python
os.environ.get("<ENV_NAME>")
```

### Step #4: Configure Web Scraper API

Bright Data의 Web Scraper API는 100개 이상의 웹사이트에서 파싱된 콘텐츠를 쉽게 가져올 수 있도록 합니다.

Web Scraper API를 설정하려면 [공식 문서](https://docs.brightdata.com/scraping-automation/web-data-apis/web-scraper-api/overview)를 참조하거나 아래 지침을 따르십시오.

아직 계정이 없다면 Bright Data 계정을 생성합니다. 로그인 후 계정 대시보드로 이동합니다. 여기에서 왼쪽의 “Web Scraper API” 버튼을 클릭합니다:

![Choosing Web Scraper API from the menu on the left](https://github.com/bright-kr/langchain-web-scraping/blob/main/Images/image-133-1024x489.png)

대상 사이트는 [CNN.com](http://cnn.com/)이므로, 검색 입력에 “cnn”을 입력하고 “CNN news — Collecy by URL” スクレイ퍼를 선택합니다:

![Searching for hte CNN Scraper API](https://github.com/bright-kr/langchain-web-scraping/blob/main/Images/image-134-1024x486.png)

현재 페이지에서 **Create token** 버튼을 클릭하여 [Bright Data API token](https://docs.brightdata.com/general/account/api-token)을 생성합니다:

![Creating a new token for the API](https://github.com/bright-kr/langchain-web-scraping/blob/main/Images/image-135-1024x408.png)

그러면 다음 모달이 열리며, 여기에서 토큰의 세부 정보를 구성할 수 있습니다:

![Configuring the details of the new token](https://github.com/bright-kr/langchain-web-scraping/blob/main/Images/image-136.png)

완료되면 **Save**를 클릭하고 Bright Data API token 값을 복사합니다.

![Once you clicked save, the new token is shown](https://github.com/bright-kr/langchain-web-scraping/blob/main/Images/image-137.png)

`.env` 파일에 아래와 같이 이 정보를 저장합니다:

```python
BRIGHT_DATA_API_TOKEN="<YOUR_BRIGHT_DATA_API_TOKEN>"
```

`<YOUR_BRIGHT_DATA_API_TOKEN>`을 모달에서 복사한 값으로 교체하십시오.

이제 CNN news Web Scraper API 페이지는 아래 예시와 비슷하게 보일 것입니다:

![The CNN Scraper API page ](https://github.com/bright-kr/langchain-web-scraping/blob/main/Images/image-138-1024x492.png)

### Step #5: Use Bright Data for Web Scraping

Web Scraper API는 요구 사항에 맞춘 작업을 시작한 다음, 스크랩된 데이터의 스냅샷을 생성합니다. 프로세스 개요는 다음과 같습니다:

1. **리クエスト 제출:** 스크랩할 페이지의 URL을 제공합니다.
2. **작업 실행:** API가 제공된 URL에서 데이터를 가져와 파싱합니다.
3. **스냅샷 조회:** 작업이 완료되면 결과를 얻기 위해 스냅샷 API를 지속적으로 쿼리합니다.

CNN Web Scraper API의 POST エンドポイント는 다음입니다:

```
"https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_lycz8783197ch4wvwg&include_errors=true"
```

해당 エンドポイント는 `url` 필드를 포함하는 객체 배열을 받고, 다음과 같은 レスポンス를 반환합니다:

```json
{"snapshot_id":"<YOUR_SNAPSHOT_ID>"}
```

이 レスポンス의 `snapshot_id`를 사용하여, 데이터를 가져오기 위해 다음 エンドポイント를 쿼리해야 합니다:

```
https://api.brightdata.com/datasets/v3/snapshot/<YOUR_SNAPSHOT_ID>?format=json
```

이 エンドポイント는 작업이 진행 중이면 HTTP 상태 코드 [`202`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/202)를 반환하고, 작업이 완료되어 데이터가 준비되면 [`200`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/200)을 반환합니다. 권장 접근 방식은 작업이 끝날 때까지 10초마다 이 エンドポイント를 폴링하는 것입니다.

작업이 완료되면 エンドポイント는 다음 형식으로 데이터를 반환합니다:

```json
[
    {
        "input": {
            "url": "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/",
            "keyword": ""
        },
        "id": "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/index.html",
        "url": "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/index.html",
        "author": "Mary Gilbert",
        "headline": "White Christmas forecast: Will you be left dreaming of snow or reveling in it?",
        "topics": [
            "weather"
        ],
        "publication_date": "2024-12-16T13:20:52.800Z",
        "updated_last": "2024-12-16T13:20:52.800Z",
        "content": "Christmas is approaching nearly as fast as Santa’s sleigh, but almost anyone in the United States fantasizing about a movie-worthy white Christmas might need to keep dreaming. Early forecasts indicate temperatures could max out around 10 to 15 degrees above normal for much of the country on Christmas Day. [omitted for brevity...]",
        "videos": null,
        "images": [
                "omitted for brevity..."
        ],
        "related_articles": [],
        "keyword": null,
        "timestamp": "2024-12-16T14:18:14.101Z"
    }
]
```

`content` 속성에는 파싱된 기사 데이터가 포함되어 있으며, 이는 여러분이 접근하려는 정보를 나타냅니다.

이를 구현하려면 먼저 `.env`에서 env를 읽고 エンドポイント URL 상수를 초기화합니다:

```
BRIGHT_DATA_API_TOKEN = os.environ.get("BRIGHT_DATA_API_TOKEN")
BRIGHT_DATA_CNN_WEB_SCRAPER_API_URL = "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_lycz8783197ch4wvwg&include_errors=true"
```

이제 위 프로세스를 재사용 가능한 함수로 변환합니다:

```python
def get_scraped_data(url):
    # Authorization headers
    headers = {
    "Authorization": f"Bearer {BRIGHT_DATA_API_TOKEN}"
    }
    # Web Scraper API payload
    data = [{
        "url": url
    }]

    # Making the POST request to the Bright Data Web Scraper API
    response = requests.post(BRIGHT_DATA_CNN_WEB_SCRAPER_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        snapshot_id = response_data.get("snapshot_id")
        if snapshot_id:
            # Iterate until the snapshot is ready
            snapshot_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"

            while True:
                snapshot_response = requests.get(snapshot_url, headers=headers)

                if snapshot_response.status_code == 200:
                    # Parse and return the snapshot data
                    snapshot_response_data = snapshot_response.json()
                    return snapshot_response_data[0].get("content")
                elif snapshot_response.status_code == 202:
                    print("Snapshot not ready yet. Retrying in 10 seconds...")
                    time.sleep(10)  # Wait for 10 seconds before retrying
                else:
                    print(f"Failed to retrieve snapshot. Status code: {snapshot_response.status_code}")
                    print(snapshot_response.text)
                    break
        else:
            print("Snapshot ID not found in the response")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
```

동작하도록 다음 두 import를 추가합니다:

```python
import requests
import time
```

### Step #6: Get Ready to Use OpenAI Models

이 예시는 LangChain 내 LLM 통합을 위해 OpenAI 모델에 의존합니다. 해당 모델을 사용하려면 환경 변수에 OpenAI API key를 구성하십시오.

기본적으로 `langchain_openai`는 [`OPENAI_API_KEY`](https://python.langchain.com/docs/integrations/llms/openai/#credentials) 환경 변수에서 OpenAI API key를 자동으로 읽습니다. 이를 설정하려면 `.env` 파일에 다음 줄을 추가합니다:

```
OPENAI_API_KEY="<YOUR_OPEN_API_KEY>"
```

`<YOUR_OPENAI_API_KEY>`를 [OpenAI API key](https://platform.openai.com/api-keys) 값으로 교체하십시오. 발급 방법을 모르면 [공식 가이드](https://platform.openai.com/docs/quickstart)를 따르십시오.

### Step #7: Generate the LLM Prompt

스크랩된 데이터를 받아 기사 요약을 얻기 위한 프롬프트를 생성하는 함수를 정의합니다:

```python
def create_summary_prompt(content, words=100):
    return f"""Summarize the following content in less than {words} words.

           CONTENT:
           '{content}'
           """
```

현재 예시에서 완전한 프롬프트는 다음과 같습니다:

```
Summarize the following content in less than 100 words.

CONTENT:
'Christmas is approaching nearly as fast as Santa’s sleigh, but almost anyone in the United States fantasizing about a movie-worthy white Christmas might need to keep dreaming. Early forecasts indicate temperatures could max out around 10 to 15 degrees above normal for much of the country on Christmas Day. It’s a forecast reminiscent of last Christmas for many, which came amid the warmest winter on record in the US. But the country could be split in two by warmth and cold in the run up to the big day. [omitted for brevity...]'
```

이를 ChatGPT에 전달하면 다음과 같이 보일 것입니다:

![Passing the task of summarizing the content in less than 100 words](https://github.com/bright-kr/langchain-web-scraping/blob/main/Images/image-139-1024x626.png)

### Step #8: Integrate OpenAI

먼저 `get_scraped_data()` 함수를 호출하여 기사 페이지에서 콘텐츠를 가져옵니다:

```python
article_url = "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/"
scraped_data = get_scraped_data(article_url)
```

`scraped_data`가 `None`이 아니라면 프롬프트를 생성합니다:

```python
if scraped_data is not None:
    prompt = create_summary_prompt(scraped_data)
```

마지막으로, [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) AI 모델로 구성된 [`ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai/) LangChain 객체에 이를 전달합니다:

```python
model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke(prompt)
```

`langchain_openai`에서 `ChatOpenAI`를 import합니다:

```python
from langchain_openai import ChatOpenAI
```

프로세스가 끝나면 `summary`에는 이전 단계에서 ChatGPT가 생성한 요약과 유사한 내용이 포함되어야 합니다:

```python
summary = response.content
```

### Step #9: Export the AI-Processed Data

이제 LangChain을 통해 선택한 AI 모델이 생성한 데이터를 사람이 읽을 수 있는 형식(예: JSON 파일)으로 내보내야 합니다.

이를 위해 원하는 데이터로 딕셔너리를 초기화한 다음, 아래와 같이 JSON 파일로 내보내고 저장합니다:

```python
export_data = {
    "url": article_url,
    "summary": summary
}

file_name = "summary.json"
with open(file_name, "w") as file:
    json.dump(export_data, file, indent=4)
```

Python 표준 라이브러리에서 [`json`](https://docs.python.org/3/library/json.html)을 import합니다:

```python
import json
```

### Step #10: Add Logs

Web Scraping AI와 ChatGPT 분석을 사용하는 スクレイピング 프로세스는 시간이 다소 걸릴 수 있습니다. 스크립트 진행 상황을 추적하기 위해, 스크립트의 핵심 단계에 `print()` 구문을 추가하여 로그를 포함합니다:

```python
article_url = "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/"
print(f"Scraping data from '{article_url}'...")
scraped_data = get_scraped_data(article_url)

if scraped_data is not None:
    print("Data successfully scraped, creating summary prompt")
    prompt = create_summary_prompt(scraped_data)

    # Ask ChatGPT to perform the task specified in the prompt
    print("Sending prompt to ChatGPT for summarization")
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(prompt)

    # Get the AI result
    summary = response.content
    print("Received summary from ChatGPT")

    # Export the produced data to JSON
    export_data = {
        "url": article_url,
        "summary": summary
    }

    print("Exporting data to JSON")
    # Write the output dictionary to JSON file
    file_name = "summary.json"
    with open(file_name, "w") as file:
        json.dump(export_data, file, indent=4)
    print(f"Data exported to '${file_name}'")
else:
    print("Scraping failed")
```

### Step #11: Put It All Together

최종 `script.py` 파일에는 다음이 포함되어야 합니다:

```python
from dotenv import load_dotenv
import os
import requests
import time
from langchain_openai import ChatOpenAI
import json

load_dotenv()

BRIGHT_DATA_API_TOKEN = os.environ.get("BRIGHT_DATA_API_TOKEN")
BRIGHT_DATA_CNN_WEB_SCRAPER_API_URL = "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_lycz8783197ch4wvwg&include_errors=true"

def get_scraped_data(url):
    # Authorization headers
    headers = {
        "Authorization": f"Bearer {BRIGHT_DATA_API_TOKEN}"
    }

    # Web Scraper API payload
    data = [{
        "url": url
    }]

    # Making the POST request to the Bright Data Web Scraper API
    response = requests.post(BRIGHT_DATA_CNN_WEB_SCRAPER_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        snapshot_id = response_data.get("snapshot_id")
        if snapshot_id:
            # Iterate until the snapshot is ready
            snapshot_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"

            while True:
                snapshot_response = requests.get(snapshot_url, headers=headers)

                if snapshot_response.status_code == 200:
                    # Parse and return the snapshot data
                    snapshot_response_data = snapshot_response.json()
                    return snapshot_response_data[0].get("content")
                elif snapshot_response.status_code == 202:
                    print("Snapshot not ready yet. Retrying in 10 seconds...")
                    time.sleep(10)  # Wait for 10 seconds before retrying
                else:
                    print(f"Failed to retrieve snapshot. Status code: {snapshot_response.status_code}")
                    print(snapshot_response.text)
                    break
        else:
            print("Snapshot ID not found in the response")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def create_summary_prompt(content, words=100):
    return f"""Summarize the following content in less than {words} words.

           CONTENT:
           '{content}'
           """

# Retrieve the content from the given web page
article_url = "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/"
scraped_data = get_scraped_data(article_url)

# Ask ChatGPT to perform the task specified in the prompt
prompt = create_summary_prompt(scraped_data)
model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke(prompt)

# Get the AI result
summary = response.content

# Export the produced data to JSON
export_data = {
    "url": article_url,
    "summary": summary
}

# Write dictionary to JSON file
with open("summary.json", "w") as file:
    json.dump(export_data, file, indent=4)
```

다음 명령으로 동작을 확인합니다:

```bash
python3 script.py
```

또는 Windows에서는:

```powershell
python script.py
```

터미널 출력은 다음과 유사해야 합니다:

```
Scraping data from 'https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/'...
Snapshot not ready yet. Retrying in 10 seconds...
Data successfully scraped, creating summary prompt
Sending prompt to ChatGPT for summarization
Received summary from ChatGPT
Exporting data to JSON
Data exported to 'summary.json'
```

프로젝트 디렉터리에 나타난 `open.json` 파일을 열면 다음과 같은 내용을 볼 수 있습니다:

```json
{
    "url": "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/",
    "summary": "As Christmas approaches, forecasts indicate temperatures in the US may be 10 to 15 degrees above normal, continuing a trend from last year\u2019s warm winter. The western US will likely remain warm, while the East experiences colder conditions leading up to Christmas. Some areas may see a mix of rain and snow, but a true \"white Christmas\" requires at least an inch of snow on the ground. Historically, cities like Minneapolis and Burlington have the best chances for snow, while places like New York City and Atlanta have significantly lower probabilities."
}
```

## Conclusion

이 접근 방식에는 몇 가지 과제가 있습니다:

- **구조 변경:** 웹사이트는 레이아웃을 자주 업데이트합니다.
- **アンチボット 조치:** 고급 방어가 일반적입니다.
- **확장성:** 대량의 데이터 추출은 복잡하고 비용이 많이 들 수 있습니다.

Bright Data의 Web Scraper API는 이러한 장애물을 극복하여, RAG 및 LangChain 기반 솔루션에 매우 유용한 도구가 됩니다.

가입하고 AI 및 LLM을 위한 추가 [offerings for AI and LLM](https://brightdata.co.kr//use-cases/data-for-ai)를 확인해 보십시오!