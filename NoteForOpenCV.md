#   OpenCV学习笔记
本笔记为为完成Robocup竞赛中KidSize组而做的OpenCV的学习

[教程](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/tutorials.html)

##  core模块，核心功能
### MAT-基本图像容器
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

[其余教程参见](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/mat%20-%20the%20basic%20image%20container/mat%20-%20the%20basic%20image%20container.html#matthebasicimagecontainer "OpenCV官网")

### 扫描图像和查找表
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
