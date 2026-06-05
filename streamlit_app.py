"""
Streamlit Community Cloud 기본 진입점.

Streamlit Cloud는 기본적으로 'streamlit_app.py'를 메인 모듈로 찾는다.
이 앱의 실제 본체는 assistant.py 이므로, 여기서 그대로 실행해 연결한다.
(Streamlit은 매 상호작용마다 이 파일을 위에서 아래로 다시 실행하므로
 runpy로 assistant.py를 매번 새로 실행해야 정상 동작한다.)
"""
import os
import runpy

_APP = os.path.join(os.path.dirname(__file__), "assistant.py")
runpy.run_path(_APP, run_name="__main__")
