'''Paddle Lite具备如下产品特色：

移动端和嵌入端的模型部署工具，可使用其部署飞桨、TensorFlow、Caffe、ONNX等多种平台的主流模型格式，包括MobileNetV1、YOLOv3、UNet、SqueezeNet等主流模型；
多种语言的API接口：C++/Java/Python，便于嵌入各种业务程序；
丰富的端侧模型：ResNet、EffcientNet、ShuffleNet、MobileNet、Unet、Face Detection、OCR_Attention等；注意因为Lite为了缩小推理库的体积，支持的算子是相对有限的，不像Paddle Inference一样支持Paddle框架的所有算子。但在移动端应用的主流轻量级模型均是支持的，可以放心使用。
支持丰富的移动和嵌入端芯片：ARM CPU、Mali GPU、Adreno GPU，昇腾&麒麟NPU，MTK NeuroPilot，RK NPU，MediaTek APU、寒武纪NPU，X86 CPU，NVIDIA GPU，FPGA等多种硬件平台；
除了Paddle Lite本身提供的性能优化策略外，还可以结合PaddleSlim可以对模型进行压缩和量化，以达到更好的性能。
Paddle Lite推理部署流程
使用Paddle Lite对模型进行推理部署的流程分两个阶段：

模型训练阶段：主要解决模型训练，利用标注数据训练出对应的模型文件。面向端侧进行模型设计时，需要考虑模型大小和计算量。
模型部署阶段：
模型转换：如果是Caffe, TensorFlow或ONNX平台训练的模型，需要使用X2Paddle工具将模型转换到飞桨的格式。
(可选步骤)模型压缩：主要优化模型大小，借助PaddleSlim提供的剪枝、量化等手段降低模型大小，以便在端上使用。
将模型部署到Paddle Lite。
在终端上通过调用Paddle Lite提供的API接口（C++、Java、Python等API接口），完成推理相关的计算。


Paddle Lite支持的模型
Paddle Lite目前已严格验证28个模型的精度和性能，对视觉类模型做到了较为充分的支持，覆盖分类、检测、分割等多个领域，包含了特色的OCR模型的支持，并在不断丰富中。其支持的list如下：

类别	类别细分	模型	支持平台
CV	分类	MobileNetV1	ARM，X86，NPU，RKNPU，APU
CV	分类	MobileNetV2	ARM，X86，NPU
CV	分类	ResNet18	ARM，NPU
CV	分类	ResNet50	ARM，X86，NPU，XPU
CV	分类	MnasNet	ARM，NPU
CV	分类	EfficientNet*	ARM
CV	分类	SqueezeNet	ARM，NPU
CV	分类	ShufflenetV2*	ARM
CV	分类	ShuffleNet	ARM
CV	分类	InceptionV4	ARM，X86，NPU
CV	分类	VGG16	ARM
CV	分类	VGG19	XPU
CV	分类	GoogleNet	ARM，X86，XPU
CV	检测	MobileNet-SSD	ARM，NPU*
CV	检测	YOLOv3-MobileNetV3	ARM，NPU*
CV	检测	Faster R-CNN	ARM
CV	检测	Mask R-CNN*	ARM
CV	分割	Deeplabv3	ARM
CV	分割	UNet	ARM
CV	人脸	FaceDetection	ARM
CV	人脸	FaceBoxes*	ARM
CV	人脸	BlazeFace*	ARM
CV	人脸	MTCNN	ARM
CV	OCR	OCR-Attention	ARM
CV	GAN	CycleGAN*	NPU
NLP	机器翻译	Transformer*	ARM，NPU*
NLP	机器翻译	BERT	XPU
NLP	语义表示	ERNIE	XPU
注意：

模型列表中 * 代表该模型链接来自PaddlePaddle/models，否则为推理模型的下载链接
支持平台列表中 NPU* 代表ARM+NPU异构计算，否则为NPU计算
Paddle Lite部署模型工作流
使用Paddle Lite部署模型包括如下步骤：

准备Paddle Lite推理库。Paddle Lite新版本发布时已提供预编译库（按照支持的硬件进行组织），因此无需进行手动编译，直接下载编译好的推理库文件即可。
生成和优化模型。先经过模型训练得到Paddle模型，该模型不能直接用于Paddle Lite部署，需先通过Paddle Lite的opt离线优化工具优化，然后得到Paddle Lite模型（.nb格式）。如果是Caffe、TensorFlow或ONNX平台训练的模型，需要先使用X2Paddle工具将模型转换到Paddle模型格式，再使用opt优化。在这一步骤中，主要会进行模型的轻量化处理，以取得更小的体积和更快的推理速度。
构建推理程序。使用前续步骤中编译出来的推理库、优化后模型文件，首先经过模型初始化，配置模型位置、线程数等参数，然后进行图像预处理，如图形转换、归一化等处理，处理好以后就可以将数据输入到模型中执行推理计算，并获得推理结果。


'''