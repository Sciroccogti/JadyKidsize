#   第15届东南大学Robocup校赛 玉华梵队
##  官方介绍
seu-unirobot-2018.zip为本次比赛所使用的webots工程

webots的基本使用及打开工程请阅读[如何使用.pdf](https://github.com/Sciroccogti/JadyKidsize/blob/master/Official%20Info/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8.pdf)

控制程序的编写请阅读[程序说明.pdf](https://github.com/Sciroccogti/JadyKidsize/blob/master/Official%20Info/%E7%A8%8B%E5%BA%8F%E8%AF%B4%E6%98%8E.pdf)

机器人官方手册[darwin-op](http://support.robotis.com/en/techsupport_eng.htm#product/darwin-op.htm)

##  总日程：
- [ ]  10月13~15日：提交代码（初测）
- [ ]  10月20日：决赛（终测）

##  实现方法
### 机器人每帧任务
1.  扫描图像
2.  规划路线
3.  移动

### 算法简介及注意点（未完待续）
* RUN：避免摔倒的条件

* LINE：二值化，噪点，区分围栏，曲线拟合，PID控制
  * 路径分为：小弯、大弯、十字路口
  * 曲线拟合：
    1.  先拟合再取中点
    2.  先取中点再拟合
  * 转弯时：
    1.  提前转头看
    2.  没路了再转头看
  * 所有跳边情况都抬头看远处，辨别路况
  * **TODO**：前后路径有拐角的十字路口
  
* BALL：找球（每一圈用不同的方式）

* 异常处理：
  * 摔倒：判定方法与解决方案

##  参数探索
头部：设为0.93（最大值）则近乎水平，设为0则大约于铅锤方向呈30°