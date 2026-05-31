# Fcllm 前端接入说明

本文档说明如何将 Rust 编写的 Qwen1.5-MoE-A2.7B 大模型后端，接入 chatgpt-web 前端页面，并以桌面 App 的形式在本地运行。

---

## 目录

- [整体架构](#整体架构)
- [第一步：准备前端页面（chatgpt-web）](#第一步准备前端页面chatgpt-web)
- [第二步：编译 Rust 后端](#第二步编译-rust-后端)
- [第三步：启动程序](#第三步启动程序)
- [启动参数说明](#启动参数说明)
- [创建桌面快捷方式](#创建桌面快捷方式)
- [接口说明（供高级用户参考）](#接口说明供高级用户参考)
- [常见问题](#常见问题)

---

## 整体架构

```
双击桌面快捷方式
        ↓
Fcllm.exe 启动
  ├── [后台] 加载 Qwen 模型权重
  ├── [后台] 启动 HTTP 推理服务（端口 8080）
  │     ├── POST /chat-process        ← chatgpt-web 前端直连
  │     ├── POST /v1/chat/completions ← OpenAI 兼容接口
  │     └── 静态文件服务              ← 托管 chatgpt-web dist/
  └── [主窗口] 弹出原生桌面窗口（WebView2）
              显示 chatgpt-web 聊天界面
```

模型推理在独立线程中运行，HTTP 服务在另一个线程，主线程负责窗口事件——互不干扰，关闭窗口即关闭整个程序。

---

## 第一步：准备前端页面（chatgpt-web）

### 1.1 下载 chatgpt-web 源码

国内网络推荐用镜像克隆（避免 GitHub 连接超时）：

```powershell
git clone https://gitclone.com/github.com/Chanzhaoyu/chatgpt-web.git E:\chatgpt-web
```

如果镜像也失败，可以用 ZIP 方式下载：

```powershell
Invoke-WebRequest -Uri "https://ghp.ci/https://github.com/Chanzhaoyu/chatgpt-web/archive/refs/heads/main.zip" -OutFile "chatgpt-web.zip"
Expand-Archive -Path "chatgpt-web.zip" -DestinationPath "E:\"
Rename-Item "E:\chatgpt-web-main" "E:\chatgpt-web"
```

### 1.2 配置 API 地址

打开 `E:\chatgpt-web\.env`，确认以下内容（已经是正确配置，无需修改）：

```env
VITE_GLOB_API_URL=http://localhost:8080
```

这一行的作用是告诉前端去 `localhost:8080` 找我们的 Rust 推理服务。

### 1.3 安装依赖（需要 Node.js 和 pnpm）

如果还没安装 pnpm：

```powershell
npm install -g pnpm
```

进入项目目录安装依赖：

```powershell
cd E:\chatgpt-web
pnpm bootstrap
```

> **如果报错 `ERR_PNPM_IGNORED_BUILDS`**，运行以下命令授权构建脚本，然后重跑 `pnpm bootstrap`：
> ```powershell
> pnpm approve-builds
> ```

### 1.4 构建前端静态文件

```powershell
cd E:\chatgpt-web
pnpm build
```

构建成功后会生成 `E:\chatgpt-web\dist\` 文件夹，里面是打包好的前端页面，后续会交给 Rust 程序来托管。

> 构建只需做一次，之后不需要重复。

---

## 第二步：编译 Rust 后端

**必须在 Visual Studio 开发者命令提示符中编译**（因为 CUDA 需要 MSVC 工具链中的 `cl.exe`）。

**打开方式**：开始菜单 → 搜索 `Developer Command Prompt for VS 2022` → 打开

```cmd
cd E:\Rust\Fcllm
cargo build --release
```

编译过程较长（首次约 10～30 分钟），编译成功后可执行文件位于：

```
E:\Rust\Fcllm\target\release\Fcllm.exe
```

> **编译只需做一次**，只有修改 Rust 代码后才需要重新编译。

---

## 第三步：启动程序

同样在 **Visual Studio 开发者命令提示符** 或普通 PowerShell 中运行：

### 方式 A：桌面原生窗口模式（推荐）

```cmd
E:\Rust\Fcllm\target\release\Fcllm.exe ^
  --server ^
  --frontend-dir "E:\chatgpt-web\dist"
```

启动过程：

1. 控制台显示模型加载进度（需要等待几分钟）
2. 显示 `模型加载完成！正在启动桌面窗口...`
3. 自动弹出一个原生 Windows 窗口，显示 chatgpt-web 聊天界面
4. 即可开始聊天

> 关闭窗口 = 退出程序（模型和 HTTP 服务一并停止）

### 方式 B：仅 HTTP 服务模式（浏览器访问）

如果不需要原生窗口，加上 `--no-ui` 参数：

```cmd
E:\Rust\Fcllm\target\release\Fcllm.exe ^
  --server ^
  --no-ui ^
  --frontend-dir "E:\chatgpt-web\dist"
```

服务启动后会自动打开默认浏览器，访问 `http://localhost:8080`。

---

## 启动参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server` | false | 启用服务器模式（接入前端） |
| `--port` | 8080 | HTTP 服务监听端口 |
| `--frontend-dir` | 无 | chatgpt-web `dist/` 文件夹路径 |
| `--no-ui` | false | 不弹窗，仅运行 HTTP 服务 |
| `--device` | cuda:0 | 推理设备（`cuda:0` / `cpu`） |
| `--max-length` | 256 | 最大生成 token 数 |
| `--memory-budget` | 0 | 显存预算（GB），0 表示不限制 |
| `--path` | model_weights/ | 模型权重根目录 |
| `--tokenizer-path` | 自动推断 | tokenizer.json 路径（通常不需要填） |

**完整示例**（自定义端口 + 限制显存）：

```cmd
Fcllm.exe --server --port 9000 --memory-budget 8 --max-length 512 --frontend-dir "E:\chatgpt-web\dist"
```

---

## 创建桌面快捷方式

程序调通之后，可以创建快捷方式，以后直接双击打开：

1. 在桌面空白处右键 → **新建** → **快捷方式**
2. 目标位置填写：
   ```
   "E:\Rust\Fcllm\target\release\Fcllm.exe" --server --frontend-dir "E:\chatgpt-web\dist"
   ```
3. 名称填 `Fcllm Chat`，点完成
4. 双击快捷方式即可启动

> **提示**：快捷方式启动时会先弹出一个黑色控制台窗口（显示模型加载进度），模型加载完毕后聊天窗口才会出现，这是正常现象。

---

## 接口说明（供高级用户参考）

程序启动后，除了桌面窗口，同时对外提供 HTTP API，可被其他工具调用。

### chatgpt-web 直连接口

```
POST http://localhost:8080/chat-process
Content-Type: application/json

{
  "prompt": "你好，请介绍一下自己",
  "systemMessage": "你是一个helpful的AI助手",
  "options": {}
}
```

响应：SSE 流式输出，每个事件包含 `{ role, id, text, delta }` 字段。

### OpenAI 兼容接口

可接入任何支持 OpenAI API 格式的第三方工具（如 Open WebUI、LobeChat 等）：

```
POST http://localhost:8080/v1/chat/completions
Content-Type: application/json
Authorization: Bearer any-key

{
  "model": "qwen1.5-moe-a2.7b",
  "messages": [
    { "role": "user", "content": "你好" }
  ],
  "stream": true
}
```

---

## 常见问题

**Q：启动后一直在加载，窗口没弹出来？**  
A：模型权重文件较大，首次加载需要几分钟，属正常现象。请耐心等待控制台显示"模型加载完成"。

**Q：编译时报 `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`？**  
A：需要在 Visual Studio 开发者命令提示符中编译，不能用普通 PowerShell。

**Q：启动时报 `无法创建 WebView（请确认 Microsoft Edge WebView2 已安装）`？**  
A：Windows 10/11 通常已预装 WebView2。若没有，请到 [Microsoft 官网](https://developer.microsoft.com/zh-cn/microsoft-edge/webview2/) 下载安装 WebView2 Runtime。使用 `--no-ui` 参数可跳过窗口，改用浏览器访问。

**Q：聊天没有回复，或者回复乱码？**  
A：检查模型权重路径是否正确，以及 tokenizer.json 文件是否存在。默认路径为：  
`E:\Rust\Fcllm\model_weights\Qwen\Qwen1.5-MoE-A2.7B\tokenizer\...\tokenizer.json`

**Q：端口 8080 被占用怎么办？**  
A：加上 `--port 其他端口号` 参数，同时修改 `E:\chatgpt-web\.env` 中的 `VITE_GLOB_API_URL` 地址，然后重新 `pnpm build`。

**Q：想换个模型怎么做？**  
A：目前代码针对 Qwen1.5-MoE-A2.7B 编写。更换模型需要修改 `src/configuration_qwen.rs` 中的模型配置。
