# DISI THE FRIST DAY USE GIT
 
## C++ C Assembly QT WINDOWSAPI LINUX STM32 ARDUINO 51 ESP32 PYTHON


### Opencv
***PLAN0001***
```
在之前结构化学习路线的基础上，为每个阶段补充对应的 OpenCV 核心图像处理算法（含 C++ 常用 API），既保留学习进度规划，又明确每个知识点对应的底层算法，适配你系统性掌握“算法+实操”的需求：

OpenCV 结构化学习路线（含核心图像处理算法）

阶段一：基础入门夯实期（1 - 2 周）

学习目标：掌握图像存储本质与基础操作，理解像素级处理逻辑

1. 核心内容+对应算法/API

◦ 图像存储基础：Mat 类（矩阵存储算法），像素遍历算法（数组下标法、指针法、迭代器法）

◦ 图像读写：imread/imwrite（多格式图像编解码算法），VideoCapture/VideoWriter（视频帧读写算法）

◦ 色彩空间转换：cvtColor（BGR/RGB/HSV 色域转换算法），split/merge（通道分离与合并算法）

◦ 简单绘制：line/rectangle/circle/putText（图形几何绘制算法）

2. 实操任务：图像格式批量转换工具、摄像头实时预览（带帧率显示）

阶段二：核心功能进阶期（2 - 3 周）

学习目标：掌握图像增强、滤波、几何变换的核心算法，解决基础图像处理问题

1. 核心内容+对应算法/API

◦ 滤波降噪算法：均值滤波（blur）、高斯滤波（GaussianBlur）、中值滤波（medianBlur）、双边滤波（bilateralFilter，保边去噪算法）

◦ 图像增强算法：全局直方图均衡化（equalizeHist）、CLAHE 自适应直方图均衡化（createCLAHE）、亮度/对比度调整（convertTo，线性变换算法）

◦ 几何变换算法：图像缩放（resize，插值算法：线性/双线性/ Lanczos）、仿射变换（warpAffine，平移/旋转矩阵算法）、透视变换（warpPerspective，单应性矩阵算法）

◦ 阈值分割算法：全局阈值（threshold，二值化/反二值化算法）、自适应阈值（adaptiveThreshold，局部阈值算法）、Otsu 阈值法（自动阈值计算算法）

2. 实操任务：老照片修复工具、文档扫描小程序（透视校正+阈值分割）

阶段三：特征提取与分割期（3 - 4 周）

学习目标：深入掌握图像特征提取与分割算法，实现目标轮廓、特征点的精准提取

1. 核心内容+对应算法/API

◦ 形态学操作算法：膨胀（dilate）、腐蚀（erode）、开运算/闭运算/梯度/顶帽（morphologyEx，形态学梯度算法）

◦ 边缘检测算法：Sobel 算子（Sobel）、Scharr 算子（Scharr）、Laplacian 算子（Laplacian）、Canny 边缘检测（Canny，多阶段边缘提取算法）

◦ 角点检测算法：Harris 角点检测（cornerHarris）、Shi - Tomasi 角点检测（goodFeaturesToTrack，改进型角点提取算法）

◦ 特征匹配算法：ORB 算法（ORB::create，特征点提取+描述子生成算法）、暴力匹配（BFMatcher::match）、FLANN 快速匹配（FlannBasedMatcher）

◦ 高级分割算法：分水岭算法（watershed，基于标记的区域分割算法）、色彩分割（inRange，HSV 色域阈值筛选算法）

2. 实操任务：物体形状识别程序（轮廓特征计算）、图像拼接功能（特征匹配+单应性矩阵）

阶段四：视频分析与目标检测期（3 - 4 周）

学习目标：掌握动态场景处理算法，实现视频目标跟踪、运动检测与目标识别

1. 核心内容+对应算法/API

◦ 视频动态分析算法：MOG2 背景建模（createBackgroundSubtractorMOG2）、KNN 背景建模（createBackgroundSubtractorKNN，运动目标提取算法）

◦ 目标跟踪算法：均值漂移（meanShift）、CamShift（CamShift，自适应窗口跟踪算法）、稀疏光流法（calcOpticalFlowPyrLK，特征点运动轨迹追踪算法）

◦ 目标检测算法：Haar 级联检测（CascadeClassifier::detectMultiScale，积分图+AdaBoost 分类算法）、模板匹配（matchTemplate，滑动窗口相似度算法：平方差/相关性匹配）

2. 实操任务：实时运动目标监控程序（背景建模+目标框选）、人脸检测小程序（Haar 级联预训练模型调用）

阶段五：综合实战与优化期（4 - 5 周）

学习目标：整合全量算法，解决复杂场景问题，优化算法性能

1. 核心内容+对应算法/API

◦ 综合算法应用：多模块算法串联（如“滤波→分割→特征匹配→目标跟踪”完整流程）

◦ 交互算法：鼠标回调（setMouseCallback，鼠标事件响应算法）、键盘控制（waitKey，按键捕获算法）

◦ 鲁棒性优化算法：图像空值判断、摄像头异常处理、算法参数自适应调整

2. 实操任务：实时锐度调节监控系统（L2 归一化+按键控制+色彩映射）、智能车牌识别 Demo（视频截取+轮廓提取+模板匹配）


```
### AI inital
***PLan0002***
```
搭建这种“本地关键词匹配+在线检索”的小型AI模型，核心是轻量化架构+模块化开发，无需深度学习训练，优先用Python实现（门槛低、库丰富，适配你的机械革命蛟龙16pro），按“基础版→进阶版→优化版”逐步迭代，具体方法和步骤如下：

一、基础版：快速搭建（无算力要求，核心实现关键词匹配+网络检索）

适合新手入门，核心依赖轻量级Python库，无需复杂算法。

1. 技术栈选择

◦ 编程语言：Python（生态完善，轻量化工具丰富）

◦ 核心库：sqlite3（本地关键词库存储）、requests（网络请求）、BeautifulSoup4（简单网页解析）、fuzzywuzzy（模糊关键词匹配）

2. 核心模块代码实现
分3个核心模块，直接拼接即可运行，算力消耗极低。

◦ 模块1：本地关键词库（用SQLite替代字典，支持批量存储和查询）

◦ 模块2：关键词匹配（模糊匹配输入词与本地库，生成检索词）

◦ 模块3：网络检索（调用搜索引擎API或轻量爬虫获取信息）

二、进阶版：增加语义关联（提升关键词匹配精度，仍轻量化）

基础版仅实现字符串匹配，进阶版可通过轻量级语义工具让关键词关联更智能，无需训练大模型。

1. 核心优化点

◦ 语义相似度计算：用sentence-transformers的轻量化预训练模型（如all-MiniLM-L6-v2，仅几MB，CPU可运行），计算输入关键词与本地关键词的语义距离，替代纯字符串匹配。

◦ 关键词扩展：调用jieba分词库对输入词分词，结合wordnet生成近义词，丰富检索词，提升查询全面性。

2. 关键代码示例（语义匹配）
from sentence_transformers import SentenceTransformer, util

# 加载轻量级语义模型（首次运行自动下载，仅几十MB）
model = SentenceTransformer('all-MiniLM-L6-v2')
# 本地关键词库
local_keywords = ["OpenCV 图像处理", "Windows API 窗口创建", "汇编语言 指令集"]
# 生成本地关键词的语义向量
local_embeddings = model.encode(local_keywords, convert_to_tensor=True)

# 输入关键词
input_keyword = "OpenCV 调整亮度"
input_embedding = model.encode(input_keyword, convert_to_tensor=True)

# 计算语义相似度，筛选Top1核心关键词
cos_scores = util.cos_sim(input_embedding, local_embeddings)
top_idx = cos_scores.argmax().item()
core_keyword = local_keywords[top_idx]
print("匹配到的核心关键词：", core_keyword)
三、优化版：增加缓存与交互逻辑（适配长期使用）

进阶版基础上，优化用户体验和运行效率，仍不增加算力负担。

1. 核心优化模块

◦ 本地缓存：用pickle或redis（轻量级）存储高频查询结果，下次相同关键词直接返回，减少网络请求。

◦ 交互界面：用tkinter或streamlit搭建简单GUI，支持输入关键词、显示结果，无需命令行操作。

◦ 错误处理：添加网络异常捕获、关键词匹配失败提示，提升模型稳定性。

四、关键工具与资源（降低开发门槛）

1. 无需训练的现成工具：直接使用LangChain的轻量化组件（如VectorStore做关键词索引、WebSearch做检索），简化模块化开发。

2. 轻量级部署：开发完成后，用PyInstaller打包成exe文件，在机械革命蛟龙16pro上双击运行，无需依赖Python环境。


```

### Metric Further
***PLAN0003***
```
你选择的 PyTorch 生态（搭配 LibTorch + OpenCV + MMDetection） ，非常适配你的 C++ 技术栈、Windows 平台，以及对“科研创新+快速迭代+复杂算法拓展”的需求。下面从 核心组件分工、Windows 平台配置、C++ 实战流程、进阶拓展方向 四个维度，提供结构化的落地指南，助力你快速上手：

一、PyTorch 生态核心组件分工（适配视觉任务全流程）

整个生态由“基础框架+视觉工具+部署接口”构成，各组件各司其职，且可无缝衔接 OpenCV，形成完整技术链：

1. LibTorch：PyTorch 的 C++ 核心库，负责模型加载、推理，提供与 Python 一致的张量运算、网络构建接口，是 C++ 项目集成 PyTorch 能力的核心。

2. TorchVision：PyTorch 官方视觉库，内置 ResNet、MobileNet 等基础模型，以及图像预处理（缩放、归一化）、数据集加载等工具，C++ 版本可直接用于基础视觉任务。

3. MMDetection：OpenMMLab 开源的目标检测工具箱，封装了 YOLOv8/v10、Faster R - CNN、Mask R - CNN 等 SOTA 算法，支持模型训练、微调，且可导出为 TorchScript 格式供 LibTorch 调用。

4. OpenCV：负责图像/视频读取（VideoCapture）、预处理（色域转换、裁剪）、推理结果可视化（绘制边界框、标注文字），弥补 PyTorch 在传统图像处理上的短板。

二、Windows 平台环境配置（关键步骤，适配机械革命蛟龙16pro）

1. 下载核心依赖

◦ 下载 LibTorch：从 PyTorch 官网 选择 Windows 系统、C++ 版本，下载预编译包（含头文件、静态库/动态库）。

◦ 下载 OpenCV：配置好已熟悉的 OpenCV 环境，确保与 LibTorch 编译架构一致（均为 x64）。

◦ 配置 MMDetection：先通过 Python 环境安装 MMDetection（用于模型训练/导出），后续将导出的模型用 LibTorch 加载。

2. VS 项目配置

◦ 包含目录：添加 LibTorch 的 include 文件夹、OpenCV 的 include 文件夹。

◦ 库目录：添加 LibTorch 的 lib 文件夹、OpenCV 的 lib 文件夹。

◦ 附加依赖项：链接 torch.lib torch_cpu.lib（LibTorch 核心库）、opencv_worldxxx.lib（OpenCV 库，xxx 为版本号）。

三、C++ 实战流程（OpenCV + LibTorch + MMDetection 目标检测示例）

核心流程：用 MMDetection 导出模型 → LibTorch 加载模型 → OpenCV 预处理图像 → 模型推理 → OpenCV 可视化结果
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
using namespace cv;
using namespace std;
using namespace torch;

int main() {
    // 1. 加载 MMDetection 导出的 TorchScript 模型（提前用 Python 导出）
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("yolov8_torchscript.pt"); // 导出的模型文件
    }
    catch (const c10::Error& e) {
        cout << "模型加载失败！" << endl;
        return -1;
    }
    module.to(torch::kCPU); // 若支持 GPU，改为 torch::kCUDA
    module.eval(); // 切换为推理模式

    // 2. OpenCV 读取并预处理图像（适配模型输入：640x640 RGB 图像）
    Mat img = imread("test.jpg");
    Mat img_resized;
    resize(img, img_resized, Size(640, 640));
    cvtColor(img_resized, img_resized, COLOR_BGR2RGB); // 转为 RGB 格式
    img_resized.convertTo(img_resized, CV_32FC3); // 转为 float32
    img_resized /= 255.0; // 归一化到 0~1

    // 3. 图像数据转为 LibTorch 张量
    auto tensor = torch::from_blob(img_resized.data, {1, 640, 640, 3}, torch::kFloat32);
    tensor = tensor.permute({0, 3, 1, 2}); // 调整维度为 [batch, channel, height, width]

    // 4. 模型推理
    vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    auto outputs = module.forward(inputs).toTensor();

    // 5. 解析输出结果（以 YOLO 目标检测为例，提取边界框和置信度）
    // 此处简化处理，实际需根据 MMDetection 模型输出格式解析
    cout << "推理输出维度：" << outputs.sizes() << endl;

    // 6. OpenCV 可视化结果（绘制边界框，实际需结合解析后的坐标）
    imshow("PyTorch + OpenCV 目标检测", img);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
四、进阶拓展方向（适配你的技术成长需求）

1. 模型优化：用 TensorRT 对 TorchScript 模型加速，提升 Windows 平台推理速度，适配实时视频处理场景。

2. 复杂任务拓展：通过 MMSegmentation 实现图像语义分割、MMTracking 实现目标跟踪，丰富项目功能。

3. Windows API 整合：用 Windows API 创建可视化窗口，将推理结果嵌入窗口，实现“图像读取 - 模型推理 - 界面展示”的完整应用。



```

### Arduino smart robot for contest

***PLAN0004***

```

结合寒假两周（14天）的时间，将每日安排拆分为上午（3小时）、下午（4小时）、晚上（2小时），节奏松紧结合，同时预留缓冲时间：

第1天：核心控制模块搭建

• 上午：

1. 拆箱整理所有配件，分类摆放（主控板/传感器/动力件）；

2. 焊接Arduino Nano 33 BLE Sense的排针（若未预装）。

• 下午：

1. 用面包板连接Nano 33 BLE Sense与UNO R4 WIFI的I2C接口；

2. 接入TP4056充电模块+电池，测试两块主板的供电稳定性（用万用表测电压）。

• 晚上：

1. 安装Arduino IDE并配置Nano 33 BLE Sense开发环境；

2. 上传“点亮板载LED”代码，验证主板功能正常。

第2天：基础外设与供电调试

• 上午：

1. 连接0.96寸OLED屏与Nano 33 BLE Sense的I2C接口；

2. 编写代码实现OLED显示“硬件测试中”，验证通信正常。

• 下午：

1. 测试电池续航（满电状态下主板+OLED持续工作时长）；

2. 整理杜邦线，用扎带固定核心模块的连线，避免杂乱。

• 晚上：

1. 学习Arduino串口通信基础，调试Nano与UNO R4的串口数据传输。

第3天：环境感知模块初装

• 上午：

1. 连接DHT11温湿度传感器与Nano 33 BLE Sense；

2. 编写代码读取温湿度数据，在OLED屏显示。

• 下午：

1. 连接HC-SR04超声波模块，编写距离检测代码；

2. 调试“距离<20cm时OLED显示‘注意避障’”的逻辑。

• 晚上：

1. 整理环境感知模块的代码，合并为一个测试程序；

2. 检查传感器接线是否牢固。

第4天：语音交互模块搭建

• 上午：

1. 连接ASRPRO语音识别模块与Nano的UART接口；

2. 烧录唤醒词固件（如“小魔”），测试语音指令识别（如“前进”）。

• 下午：

1. 连接UNV TTS语音合成模块，调试文字转语音功能；

2. 实现“唤醒→说‘温度多少’→TTS播报当前温度”的流程。

• 晚上：

1. 优化语音模块的串口占用冲突（若与传感器共用串口）；

2. 记录语音指令的识别准确率，调整麦克风距离。

第5天：声源定位模块调试

• 上午：

1. 连接4麦环形阵列（AR-1105）与UNO R4 WIFI的I2C接口；

2. 烧录阵列的声源定位固件，测试串口输出方位角数据。

• 下午：

1. 编写代码将方位角转换为“东/南/西/北”，在OLED屏显示；

2. 测试不同方向的声音（拍手），验证定位准确性。

• 晚上：

1. 固定麦克风阵列的位置（建议安装在机器人顶部）；

2. 补充顶部单麦克风的接线（提前到货的话）。

第6天：动力系统硬件组装

• 上午：

1. 组装履带底盘的支架与轮子；

2. 安装130电机到底盘，连接电机线到L298N驱动板。

• 下午：

1. 连接L298N与UNO R4 WIFI的PWM接口；

2. 编写代码实现“前进/后退/左转/右转”的电机控制。

• 晚上：

1. 测试电机转速与转向，调整PWM占空比优化动力；

2. 固定驱动板到底盘，整理电机线。

第7天：动力+避障联动调试

• 上午：

1. 整合HC-SR04与L298N的代码，实现“距离<15cm自动后退”；

2. 测试避障功能的响应速度，调整传感器采样频率。

• 下午：

1. 调试“语音指令‘前进’→电机启动+避障生效”的联动；

2. 解决电机启动时的电源波动问题（可增加电容滤波）。

• 晚上：

1. 整理动力模块的代码，添加异常保护逻辑（如电机堵转自动停止）；

2. 简单清洁底盘的组装碎屑。

第8天：视觉模块硬件连接

• 上午：

1. 连接OV2640摄像头与ESP32-S3的CSI接口；

2. 烧录ESP32-S3的摄像头驱动固件，测试图像采集功能。

• 下午：

1. 部署OpenCV人脸检测库到ESP32-S3；

2. 调试“识别到人脸→串口输出‘检测到人’”的功能。

• 晚上：

1. 测试不同光线环境下的人脸识别准确率；

2. 固定摄像头到舵机云台支架。

第9天：舵机云台与视觉联动

• 上午：

1. 连接SG90舵机与Nano 33 BLE Sense的PWM接口；

2. 编写代码实现舵机水平/俯仰转动（0-180°）。

• 下午：

1. 整合声源定位与舵机代码，实现“方位角→舵机转向对应方向”；

2. 测试“声源定位→舵机转动→摄像头对准声源”的流程。

• 晚上：

1. 调整舵机转动速度，避免卡顿；

2. 补充顶部麦克风的信号对比代码（若已到货）。

第10天：顶部声源与俯仰功能实现

• 上午：

1. 安装顶部麦克风到摄像头上方，连接到UNO R4的ADC接口；

2. 编写代码读取顶部麦克风与环形阵列的信号强度。

• 下午：

1. 实现“顶部信号>环形阵列→舵机俯仰抬头”的逻辑；

2. 测试“上方拍手→摄像头抬头+语音问候”的互动。

• 晚上：

1. 优化信号对比的阈值（避免误触发）；

2. 录制俯仰功能的测试视频。

第11天：全功能联动调试

• 上午：

1. 整合所有模块代码：语音唤醒→声源定位→舵机转向→人脸识别→语音回复；

2. 测试完整流程的连贯性（如“唤醒→说‘你好’→定位→转向→识别→回复‘你好呀’”）。

• 下午：

1. 解决模块间的代码冲突（如串口占用、变量命名重复）；

2. 优化各功能的响应延迟（目标：从唤醒到回复<3秒）。

• 晚上：

1. 记录当前BUG（如偶发的舵机卡顿），列出修复清单；

2. 备份所有代码到SD卡。

第12天：BUG修复与优化

• 上午：

1. 修复舵机卡顿问题（检查供电或PWM信号稳定性）；

2. 优化语音识别的抗干扰性（调整麦克风灵敏度）。

• 下午：

1. 测试复杂环境（如宿舍多人说话）下的功能稳定性；

2. 调整避障距离阈值，避免误触发。

• 晚上：

1. 完善代码注释，便于后续修改；

2. 整理硬件连接示意图（拍照+标注引脚）。

第13天：功能验收与细节优化

• 上午：

1. 全功能测试3次：语音控制、避障、人脸+声源联动；

2. 补充“低电量时OLED显示提醒”的功能。

• 下午：

1. 美化机器人外观（用贴纸装饰底盘、整理外露连线）；

2. 录制完整的功能演示视频（时长3-5分钟）。

• 晚上：

1. 整理项目材料：代码、硬件清单、演示视频；

2. 准备第二天的展示内容。

第14天：项目收尾与展示

• 上午：

1. 最终测试所有功能，确保无BUG；

2. 编写项目说明文档（包含功能介绍、组装步骤）。

• 下午：

1. 向家人/朋友展示机器人功能；

2. 备份所有项目文件到电脑+U盘。

• 晚上：

1. 总结项目收获，记录待改进的点（如增加更多语音指令）；

2. 整理工具与剩余配件，妥善收纳。


```
## FINISHED PROJECT
***千里之行，始于足下，不积跬步，无以至千里***
1. **Code syntax**
``` 
1. C++
2. C
3. 8086 Assembly
```
2. **Library function**
```
1. EasyX

```
3. **Tools**
```

```
4. **Enbed**
```

```