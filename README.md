# PyStart
This project serves as an example for a project using DevContainers

# Setup
Start by forking the project 
![](./docs/images/project-fork.png?raw=true)

Once the project is in your account, there are different ways you can set up this project. We will cover how you can set it up in [GitHub Codespaces](https://github.com/features/codespaces) and in VS Code on your local machine.

## GitHub Codespaces
You can set up this project to develop in [GitHub Codespaces](https://github.com/features/codespaces), where you can code, debug, and run your app remotely in a codespace. A codespace provides a fully configured development environment hosted in the cloud, eliminating the need for local setup. This environment includes your project's dependencies, tools, and extensions, ensuring a consistent and reproducible development experience. It streamlines collaboration by providing real-time editing, integrated version control, and easy access to debugging and testing tools, all while maintaining the security and reliability of your project.

Steps:

1. Click on the "<> Code" button
2. Click on the "Codespaces" tab
3. Click on the "Create codespace on main"

![](./docs/images/pystart.png?raw=true)


## Locally in VS Code
You first need to set up your Python development environment. Specifically, this tutorial requires:

Python 3.11 (check the installation guide if you don't have it installed)
Python extension for VS Code (For additional details on installing extensions, you can read Extension Marketplace).

It is recommended to use conda or env to isolate the python environment:
- [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) information
- [venv](https://docs.python.org/3/library/venv.html) information

VS Code should have the following extensions:
- [ms-python.python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [ms-python.vscode-pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [esbenp.prettier-vscode](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)
- [ms-python.black-formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
- [charliermarsh.ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
- [ms-python.debugpy](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)


## VS Code and DevContainer
If you prefer to use DevContainer, you just need to install the DevContainer extension and all necessary extensions will be taken care for you.
Follow the instructions from the [DevContainer site](https://code.visualstudio.com/docs/devcontainers/tutorial).

# Running in Codespaces
To run the FastAPI server, run the following command below. This command will reload the server as you make changes to the code.
```
uvicorn main:app --reload
```

Once you run it, you will see a pop-up window that will make your application available to the outside. Click on "Open in browser" button to open in your browser:
![](./docs/images/running.png?raw=true)

If the pop-up dissapear before you can click on it, you can click on the "PORTS" tab nad click on the "Open in browser" button as you hover the URL as shown below:
![](./docs/images/running-manually.png?raw=true)

After you enable the port to be access via your browser, you will see your browser open to the application. You can now use the URL the browser points to for your API access.
![](./docs/images/running-browser.png?raw=true)