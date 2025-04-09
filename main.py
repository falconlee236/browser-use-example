
from browser_use import Agent, Browser, BrowserConfig, SystemPrompt
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
from overrides import overrides

from langchain_core.messages import SystemMessage
class MySystemPrompt(SystemPrompt):
    @overrides
    def get_system_message(self) -> SystemMessage:
        
        # Get existing rules from parent class
        existing_prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

        # Add your custom rules
        new_prompt = """
            10. MOST IMPORTANT RULE:
            - You must response, think, reason Only Korean
            - You must find "ReactModal__Overlay" attribute when you search sections
            - You must search another apporach, another section, another actions even you faced retry the situations over twice times
        """

        # Make sure to use this pattern otherwise the exiting rules will be lost
        return SystemMessage(content=f'{existing_prompt}\n{new_prompt}')

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
# planner_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
planner_llm = ChatOpenAI(model="gpt-4o-mini")

log_dir_path = "./"
browser = Browser(
    config=BrowserConfig(
        # chrome_instance_path="/Applications/Arc.app/Contents/MacOS/Arc",
        headless=False,
		disable_security=True,
    )
)
browser_context_config = BrowserContextConfig(
	wait_for_network_idle_page_load_time=3.0,
    browser_window_size={'width': 1480, 'height': 1100},
    locale='ko-KR',
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    # viewport_expansion=500,
    viewport_expansion=-1, # 이게 html 속성 개수 보는 너비를 의미함
    allowed_domains=["korail.com"], # 허용된 도메인만 추가로 init_action으로 추가 탭을 열 수 있다.
)

context = BrowserContext(
    browser=browser,
    config=browser_context_config
)

inital_actions = [
    {
        'open_tab': {
            "url": "https://www.korail.com/ticket/main"
        },
    },
    {
        'scroll_down': {
            'amount': 100
        }
    },
]


# Define sensitive data
# The model will only see the keys (x_name, x_password) but never the actual values
sensitive_data = {
    'x_number': os.environ.get("PHONE_NUMBER"), 
    'x_password': os.environ.get("PASSWORD")
}

# 여기에 민감 데이터 key값이 들어가야한다.
task = """
2명이 타야하는 기차표를 예매해줘
그러기 위해서는 로그인을 하고 열차를 조회한다음에 좌석을 찾아야겠지?
아니 그리고 날짜가 없으면 화살표나 스크롤을 하면서 찾으면 되는것도 알지?
예약 다하면 아래에 있는 예약 버튼을 눌러서 예약 확인 페이지까지 보여주면 완료야


<중요 정보>
그리고 로그인을 한다면 휴대폰 번호로 로그인할거야. 정보는 다음과 같아
전화번호 = x_number \
비밀번호 = x_password \
    
조건은 다음과 같아
날짜 = 2025년 5월 1일
출발역 = 서울
도착역 = 부산
총 인원수 = 2명

<참고사항>
나는 창가자리가 좋지만 창가 자리가 없으면 상관 없어.
하지만 두 좌석이 붙어 있어야해.
시각은 저녁시간대면 좋겠고 그런 기차표가 없다면 다른 날짜를 찾아서라도 구해와.
제발 overlay된 html 속성을 집중하고, 그게 어려우면 먼저 예약 버튼을 누르고 설정하자.
"""

async def main():
    agent = Agent(
        task=task,
        llm=llm,
        use_vision=True,
        save_conversation_path=log_dir_path + "/conversation",
        browser=browser,
        browser_context=context,
        initial_actions=inital_actions,
        planner_llm=planner_llm, # Separate model for planning
        use_vision_for_planner=True, # Disable vision for planner
        planner_interval=4, # Plan every 4 steps
        system_prompt_class=MySystemPrompt,
        sensitive_data=sensitive_data,
    )
    history = await agent.run(
        max_steps=500,
    )
    await browser.close()
    # Access (some) useful information
    print(history.final_result())



from lmnr import Laminar
Laminar.initialize(
    project_api_key=os.environ.get("LMNR_PROJECT_API_KEY"),
)


if __name__ == "__main__":
    for index in range(1, 100):
        log_dir_path = f"./logs/{index}"
        if os.path.isdir(log_dir_path) is False:
            break
    asyncio.run(main())


