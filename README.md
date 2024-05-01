# MLVU-project
SNU 2024-1 MLVU Project

## Requirements
- Python 3.10 (or higher)
- [Poetry](https://python-poetry.org/)

## Conventions

### Active Branches
- `main`(default 브랜치): 최종적인 결과물 저장하는 브랜치
- `dev`: 개발용 중간 브랜치(여러 기능들 합치는 브랜치)
- `feature/{description}`: 새로운 기능이 추가되는 경우에 사용
- `refactor/{description}`: 기능 변경 없이 코드 리팩토링만을 하는 경우에 사용
- `fix/{description}`: `dev` 브랜치로 반영하는 사소한 오류 수정 시에 사용
- `hotfix/{description}`: `prod` 브랜치로 반영하는 긴급한 오류 수정 시에 사용

### PR Merge Rules
  - default: *Squash and merge*
  - `dev` to `main`: *Create a merge commit*

### Code Styles
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Ruff](https://docs.astral.sh/ruff/)를 이용하여 코드 스타일을 검사합니다. 현재 세팅은 diffusers 라이브러리와 싱크를 맞춰두었습니다. 각자 IDE에서 잘 동하는지 확인해보시고, 문제가 있으면 말씀해주세요!

```shell
# Check code styles using ruff
ruff check --fix
```
등의 명령어를 사용해보시면 됩니다.

## Dev Guidelines

### Python Dependencies
가상환경을 활성화하고 필요한 패키지를 설치합니다.
```shell
poetry shell
poetry install
```
`pyproject.toml` 파일의 패키지 목록을 변경한 경우, 아래 명령을 통해 `poetry.lock` 파일을 최신화합니다.
```shell
poetry lock
```
