#   [OpenCV学习笔记](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/tutorials.html)
本笔记为为完成Robocup竞赛中KidSize组而做的OpenCV的学习

**笔记中各标题均为原教程链接**

##  [core模块，核心功能](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/table_of_content_core/table_of_content_core.html)
### [MAT-基本图像容器](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/mat%20-%20the%20basic%20image%20container/mat%20-%20the%20basic%20image%20container.html)
*Mat*是一个类，由两个数据部分组成：
1.  **矩阵头**（包含矩阵尺寸，存储方法，存储地址等信息），又称**信息头**；
2.  **矩阵指针**，一个指向存储所有像素值的矩阵（根据所选存储方法的不同矩阵可以是不同的维数）的指针，为**常数**。

在OpenCV中，每个*Mat*有自己的信息头，但共享同一个矩阵（即矩阵指针指向同一地址）。故复制构造只复制信息头和矩阵指针。
```C++
Mat A, C;                                 // 只创建信息头部分
A = imread(argv[1], CV_LOAD_IMAGE_COLOR); // 这里为矩阵开辟内存

Mat B(A);                                 // 使用拷贝构造函数

C = A;                                    // 赋值运算符
```
另外，你还可以创建只引用部分数据的信息头：
```C++
Mat D (A, Rect(10, 10, 100, 100) ); //引入一个矩形
Mat E = A(Range:all(), Range(1,3)); //只引用边界参数
```
使用成员函数`clone()`或`copyTo()`来复制包括矩阵数据、信息头和矩阵指针的矩阵

*Mat*支持流式输出，但仅限二维矩阵

**构造函数**：
*   `Mat M(行数, 列数, CV_8UC3, Scalar(0,0,255)); `或`Mat M(维数, 各维尺寸矩阵, CV_8UC(1), Scalar(0,0,0))`
*   `CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]`
*   初始化一个纯白矩阵可用`Scalar::all(0)`

其余部分参见原教程

### [扫描图像和查找表](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html)
![](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/d741bf066ad4641450e60523988450519478814d.png)

灰度图像矩阵

![](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/154cd030d6aa7d29c35852ac468a3e3e14b882bf.png)

RGB颜色模型的矩阵

**子列的通道顺序为BGR而非RGB！**

推荐的查找表赋值方法：
1.  高性能法
```C++
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));     
    int channels = I.channels();
    int nRows = I.rows * channels; 
    int nCols = I.cols;

    if (I.isContinuous())  // make sure the memory of the Mat is continuous
    {
        nCols *= nRows;
        nRows = 1;         
    }

    int i,j;
    uchar* p; 
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);  // get the pointer of the beginning of the row
        for ( j = 0; j < nCols; ++j)  // scan til the end of the row
        {
            p[j] = table[p[j]];             
        }
    }
    return I; 
}
```
或使用*data*
```C++
uchar* p = I.data;

for( unsigned int i = 0; i < ncol*nrows; ++i)
    *p++ = table[*p];
```
2.  迭代法
```C++
Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));     
    
    const int channels = I.channels();  // figure out if the Mat is grey or coloourful
    switch(channels)
    {
    case 1: 
        {
            MatIterator_<uchar> it, end; 
            for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
                *it = table[*it];
            break;
        }
    case 3: 
        {
            MatIterator_<Vec3b> it, end; 
            for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
            {
                (*it)[0] = table[(*it)[0]];
                (*it)[1] = table[(*it)[1]];
                (*it)[2] = table[(*it)[2]];
            }
        }
    }
    
    return I; 
}
```
3.  返回某像素点数据的On-the-fly地址计算，即`at()`（以下为灰度的实例）
```C++
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));     

    const int channels = I.channels();
    switch(channels)
    {
    case 1: 
        {
            for( int i = 0; i < I.rows; ++i)
                for( int j = 0; j < I.cols; ++j )
                    I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];
            break;
        }
    case 3: 
        {
         Mat_<Vec3b> _I = I;
            
         for( int i = 0; i < I.rows; ++i)
            for( int j = 0; j < I.cols; ++j )
               {
                   _I(i,j)[0] = table[_I(i,j)[0]];
                   _I(i,j)[1] = table[_I(i,j)[1]];
                   _I(i,j)[2] = table[_I(i,j)[2]];
            }
         I = _I;
         break;
        }
    }
    
    return I;
}
```
4.  核心函数LUT（最被推荐用于实现批量图像元素查找和更改操作图像方法）
```C++
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.data; 
    for( int i = 0; i < 256; ++i)
        p[i] = table[i];

    LUT(I, lookUpTable, J);  // "I" is input, "J" is output
```

**为得到最优速度，最好尝试四种算法后比较**

**建议：尽量使用OpenCV内置函数，尤其是和Intel PY的LUT函数：)**

### [矩阵的掩码操作](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/mat-mask-operations/mat-mask-operations.html)
又称：**掩膜**，培训PPT强调内容！

根据掩码矩阵（也作：*核*）重新计算图像中每个像素的值。掩码矩阵中的值表示近邻像素值（包括该像素自身的值）对新像素值有**多大影响**。从数学观点看，我们用自己设置的权值，对像素邻域内的值做了个**加权平均**。

例如：增强图像对比度：
1.  将掩码矩阵中心元素对齐到要计算的目标像素上；
2.  将邻域像素值与相应的掩码矩阵元素值的乘积相加

1.  基本方法
```C++
void Sharpen(const Mat& myImage,Mat& Result)
{
    CV_Assert(myImage.depth() == CV_8U);  // 异常处理：确保仅接受uchar图像

    Result.create(myImage.size(),myImage.type());  // 初始化输出矩阵
    const int nChannels = myImage.channels();

    for(int j = 1 ; j < myImage.rows-1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);  // 获取当前行的前一行的指针
        const uchar* current  = myImage.ptr<uchar>(j    );  // 获取当前行的指针
        const uchar* next     = myImage.ptr<uchar>(j + 1);  // 获取当前行的后一行的指针

        uchar* output = Result.ptr<uchar>(j);  // 输出矩阵的指针

        for(int i= nChannels;i < nChannels*(myImage.cols-1); ++i)
        {
            *output++ = saturate_cast<uchar>(5*current[i] - current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);  // 每次输出后后移输出指针
        }
    }

    // 以下将边界点直接置零（因为边界对比度计算时引用了图像外的数据）
    Result.row(0).setTo(Scalar(0));  // 上边界
    Result.row(Result.rows-1).setTo(Scalar(0));  // 下边界
    Result.col(0).setTo(Scalar(0));  // 左边界
    Result.col(Result.cols-1).setTo(Scalar(0));  // 有边界
}
```
2.  使用滤波器掩码函数
```C++
Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
                               -1,  5, -1,
                                0, -1,  0);  // 掩码矩阵

filter2D(I, K, I.depth(), kern);  // 参数分别为：输入图像，输出图像，掩码（，核的中心，在未定义区域的行为）
```

### [图像求和](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/adding_images/adding_images.html)
略

### [更改对比度与亮度](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/basic_linear_transform/basic_linear_transform.html)
*图像处理算子*是带有一幅或多幅输入图像，产生一幅输出图像的函数

图像变换分为：
*   点算子（像素变换）
*   邻域算子（基于区域）

**点算子**：
*   在这一类图像处理变换中，仅仅根据输入像素值（有时可加上某些全局信息或参数）计算相应的输出像素值；
*   这类算子包括*亮度*和*对比度*调整 ，以及*颜色校正和变换*。

**亮度和对比度调整**常用常数对点进行*乘法*和*加法*：$r(x)={\alpha}e(x)+\beta$，其中α>0与β称作*增益*和*偏置*参数，分别用于控制*对比度*和*亮度*
```C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;  // 危险！！
using namespace cv;  // 危险！！

double alpha;  // 控制对比度
int beta;  // 控制亮度

int main( int argc, char** argv )
{
    // 读入用户提供的图像
    Mat image = imread( argv[1] );
    Mat new_image = Mat::zeros( image.size(), image.type() );  // Matlab风格初始化，使图像有原图像的大小与类型，且为空

    // 初始化
    cin >> alpha;  // alpha应在1.0到3.0之间
    cin >> beta;  // beta应在0到100之间

    // 执行运算 new_image(i,j) = alpha*image(i,j) + beta
    for( int y = 0; y < image.rows; y++ )
    {
        for( int x = 0; x < image.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )  // 分别访问R、G、B
            {
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );  // saturate_cast用于确保结果未超出取值范围，且为整数
            }
        }
    }

    // 创建窗口
    namedWindow("Original Image", 1);
    namedWindow("New Image", 1);

    // 显示图像
    imshow("Original Image", image);
    imshow("New Image", new_image);

    return 0;
}
```
也可使用`image.convertTo(new_image, -1, alpha, beta);`

### [基本绘图](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/basic_geometric_drawing/basic_geometric_drawing.html)
略

### [随机数发生器与绘制文字](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/random_generator_and_text/random_generator_and_text.html)
略

### [离散傅里叶变换](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html)
用于将图像从空间域转换到频域，常用于决定图片中物体的几何方向

略

### [输入输出XML和YAML文件](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/file_input_output_with_xml_yml/file_input_output_with_xml_yml.html)
略

### [与OpenCV 1 同时使用](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.html)
略

##  [imgproc模块：图像处理](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/table_of_content_imgproc/table_of_content_imgproc.html)
### [图形平滑处理](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html#smoothing)
*平滑*又称*模糊*，可用于减少噪声等，需使用滤波器

*线性滤波器*最为常用，输出的像素值为输入像素值的加权和

不妨把*滤波器*想象为一个包含加权系数的窗口，当使用这个滤波器平滑处理图像时，就把这个窗口滑过图像。

**归一化快滤波器**：
*   算术平均，即各像素权值相等
*   `blur(src, dst, Size(w, h), Point(-1, -1))`函数，参数为输入图像，输出图像，内核大小，锚点（被平滑点）位置（为负则取核的中心）

**高斯滤波器**：
*   将输入数组的各像素与*高斯内核*卷积，并将卷积和输出
*   二位高斯函数：$G_0(x,y)=Ae\frac{-(x-\mu_x)^2}{2\delta^2_x}+\frac{-(y-\mu_y)^2}{2\delta^2_y}$，其中μ为均值，δ为标准差
*   `GaussianBlur(src, dst, Size(w, h), deltaX, deltaY)`，参数为输入图像，输出图像，内核大小（w与h须为正奇数，否则将用两个δ来计算内核大小），x、y方向标准方差

**中值滤波器**：
*   将每个像素用邻域（以当前像素为中心的正方形区域）像素的中值代替
*   `medianBlur(src, dst, i)`函数，参数为输入图像，输出图像（需与src相同类型），内核大小（须为奇数）

**双边滤波器**：
*   避免将边缘磨掉
*   每个邻域像素都有一个权值，权值分为两部分：
    1.  第一部分与高斯滤波一样
    2.  第二部分取决于该邻域像素与当前像素的灰度差值
*   `bilateralFilter(src, dst, d, deltaC, deltaS)`函数，参数为输入图像，输出图像，邻域直径，颜色空间的标准差，坐标空间的标准差（单位：像素）

其余部分参见原教程

### [腐蚀与膨胀](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html)
**开闭操作即基于此！**培训PPT强调内容！

**形态学操作**：基于形状的一系列图像处理操作。通过将*结构元素*作用于输入图像来产生输出图像。

*腐蚀与膨胀*是最基本的形态学操作，常用于
1.  除噪
2.  分割独立的图像元素，或连接相邻的元素
3.  寻找图片中的明显的极大值区域或极小值区域

**膨胀**：
1.  将图像A与任意形状的内核B（通常为正方形或圆形）进行卷积
2.  内核B有一个可定义的*锚点*，通常为内核中心
3.  进行膨胀操作时，将内核B划过图像,将内核B覆盖区域的最大像素值提取，并代替锚点位置的像素。显然，这一最大化操作将会导致图像中的*亮区*开始膨胀
4.  其余参考腐蚀
```C++
void Dilation( int, void* )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// 膨胀操作
  dilate( src, dilation_dst, element );
  imshow( "Dilation Demo", dilation_dst );
}
```

**腐蚀**：
1.  与膨胀恰恰相反，使*亮区*被腐蚀
2.  `erode(src, erosion_dst, element)`函数，参数为输入图像，输出图像，内核（默认为3*3矩阵）
3.  内核自定义：使用函数`getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size, erosion_size ) );`指定内核为矩形：MORPH_RECT，交叉形：MORPH_CROSS，椭圆形：MORPH_ELLIPSE
```C++
void Erosion( int, void* )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
  /// 腐蚀操作
  erode( src, erosion_dst, element );
  imshow( "Erosion Demo", erosion_dst );
}
```

