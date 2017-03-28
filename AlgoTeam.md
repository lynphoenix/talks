# Projects

## 工程项目
* 纵向项目
  * 鉴黄
    * 评估数据集
        提交模型，指定数据集 -> 返回结果
        模型：prototxt，caffemodel
        数据集：nfs/json list
        返回的结果做分析
    * 评估
    * 打标后处理
      * 输入：markjson文件zip
      * 输出：json in bucket

  * 暴恐
    * 评估数据集
    * 打标
    * 评估

  * 通用检测
  * 镜头分割
  * 跟踪

* 横向项目
  * 人脸聚类 相似度
  * 装修图片分类
  * 卡片识别
  * 结构化OCR项目 - 身份证，营业执照，银行卡，名片等

* 平台需求
  * 分布式训练
  * 训练过程信息可视化
  * 数据接口规整
  * 数据Team管理

* Demo 项目 - 通用化，配套训练，数据接口规范化
  * DeepVideoAnalytics
    * 基于NN算法的visual search
    * 上传视频、图片包
    * 用youtube-dl下载youtube url
    * 在Postgres中存储metadata
    * 基于celery和rabbitMQ执行 query，frame extraction，indexing等操作
    * Videos, frames, indexes, numpy vectors stored in media directory, served through nginx

  * Classification - 模型可替换，类别可替换
  * Detection - 模型可替换，类别可替换
  * Segmentation
  * Image Captioning
  * Video Captioning
  * Video Detection & Tracking
  * 人流统计


## 中期研究项目 - 半年

* 通用检测
  * Demos
    * YOLO
    * craftGBD
    * 检测 + 跟踪
  * 评估系统
    * 系统搭建
    * 数据采集
  * 手机SDK


* 图像/视频语义描述

* 分割

* 人流统计

* GAN积累
  * 样本集放大
  * 超分辨率
  * pix2pix
  * 人像复原
  * 图像上色
  * 特征表达
  * 图像生成

  * 风格画



## 长期研究项目 - 一年

* 七牛图片信息利用
* 细粒度特征识别
* 图片/视频检索
* 图片结合推荐 - wide & deep
* 知识图谱建立

## Google API

### Vision API
* Face
  * Detection
  * Alignment
  * 表情
  * 情绪
  * 是否模糊
  * 头饰
  * confidence

* Label
  * 人
  * 男/女
  * 裸胸
  * 裸体
  * 手臂，手指，嘴，胸，腹部
  * 自拍

* Web
  * 包含匹配图片的页面
  * 完全匹配的图片
  * 部分匹配的图片

* Content
  * 成人图片
  * 是否有诈骗
  * 暴力
  * medical

* Landmark
* Logo
* OCR

### Video API
* video catalog
* 从噪声中提取有效信息
* 视频中的图像标签
* 场景切换
* 焦点区域监控
