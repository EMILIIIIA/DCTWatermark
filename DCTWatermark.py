import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


class DCT_Embed(object):
    def __init__(self, backgroundurl, watermarkurl, block_size=8, alpha=30,output=""):

        # 读取背景

        background = cv2.imread(backgroundurl)
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)


        # 读取水印
        watermark=cv2.imread(watermarkurl, cv2.IMREAD_GRAYSCALE)
        # 处理水印长宽
        bgh=self.background.shape[0]//block_size
        bgl=self.background.shape[1]//block_size
        
        if watermark.shape[0]//bgh>=watermark.shape[1]//bgl:
            watermark = cv2.resize( watermark,(watermark.shape[1]//(watermark.shape[0]//bgh),bgh))
        else:
            watermark = cv2.resize( watermark,(bgl,watermark.shape[0]//(watermark.shape[1]//bgl)))
        # watermark二值化
        self.watermark = np.where(watermark < np.mean(watermark), 0, 1)  
        # 将RBG格式的背景转为YUV格式，Y为灰度层，U\V为色彩层，此处选择U层进行嵌入
        self.yuv_background = cv2.cvtColor(self.background, cv2.COLOR_RGB2YUV)  
        Y, U, V = self.yuv_background[..., 0], self.yuv_background[..., 1], self.yuv_background[..., 2]
        self.bk = U  # 嵌入对象为bk

        b_h, b_w = background.shape[:2]
        w_h, w_w = watermark.shape[:2]

        # 保存参数
        self.block_size = block_size
        # 水印强度控制
        self.alpha = alpha
        # 随机的序列
        self.k1 = np.random.randn(block_size)
        self.k2 = np.random.randn(block_size)


    def run(self):
        # 进行分块与DCT变换
        background_dct_blocks = self.dct_blkproc(background=self.bk)  # 得到分块的DCTblocks

        # 嵌入水印图像
        embed_watermak_blocks = self.dct_embed(dct_data=background_dct_blocks, watermark=self.watermark)  # 在dct块中嵌入水印图像

        # 将图像转换为空域形式
        synthesis = self.idct_embed(dct_data=embed_watermak_blocks)  # idct变换得到空域图像
        self.yuv_background[..., 1] = synthesis
        rbg_synthesis = cv2.cvtColor(self.yuv_background, cv2.COLOR_YUV2RGB)
        # 提取水印
        extract_watermark = self.dct_extract(synthesis=synthesis, watermark_size=self.watermark.shape) * 255
        extract_watermark.astype(np.uint8)
        # 文件保存
        cv2.imwrite(output,cv2.cvtColor(rbg_synthesis, cv2.COLOR_RGB2BGR))
        # 可视化处理
        images = [self.background ,self.watermark, rbg_synthesis, extract_watermark]
        titles = ["background", "watermark", "systhesis", "extract"]
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            if i%2:
                plt.imshow(images[i],cmap=plt.cm.gray)
            else:
                plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis("off")
        plt.show()



    def dct_blkproc(self, background):
        """
        对background进行分块，然后进行dct变换，得到dct变换后的矩阵

        :param image: 输入图像
        :param split_w: 分割的每个patch的w
        :param split_h: 分割的每个patch的h
        :return: 经dct变换的分块矩阵、原始的分块矩阵
        """
        background_dct_blocks_h = background.shape[0] // self.block_size  # 高度
        background_dct_blocks_w = background.shape[1] // self.block_size  # 宽度
        background_dct_blocks = np.zeros(shape=(
            (background_dct_blocks_h, background_dct_blocks_w, self.block_size, self.block_size)
        ))  # 前2个维度用来遍历所有block，后2个维度用来存储每个block的DCT变换的值

        h_data = np.vsplit(background, background_dct_blocks_h)  # 垂直方向分成background_dct_blocks_h个块
        for h in range(background_dct_blocks_h):
            block_data = np.hsplit(h_data[h], background_dct_blocks_w)  # 水平方向分成background_dct_blocks_w个块
            for w in range(background_dct_blocks_w):
                a_block = block_data[w]
                background_dct_blocks[h, w, ...] = cv2.dct(a_block.astype(np.float64))  # dct变换
        return background_dct_blocks

    def dct_embed(self, dct_data, watermark):
        """
        将水印嵌入到载体的dct系数中
        :param dct_data: 背景图像（载体）的DCT系数
        :param watermark: 归一化二值图像0-1 (uint8类型)
        :return: 空域图像
        """
        temp = watermark.flatten()
        assert temp.max() == 1 and temp.min() == 0, "为方便处理，请保证输入的watermark是被二值归一化的"

        result = dct_data.copy()
        for h in range(watermark.shape[0]):
            for w in range(watermark.shape[1]):
                k = self.k1 if watermark[h, w] == 1 else self.k2
                # 查询块(h,w)并遍历对应块的中频系数（主对角线），进行修改
                for i in range(self.block_size):
                    result[h, w, i, self.block_size - 1] = dct_data[h, w, i, self.block_size - 1] + self.alpha * k[i]
        return result

    def idct_embed(self, dct_data):
        """
        进行对dct矩阵进行idct变换，完成从频域到空域的变换
        :param dct_data: 频域数据
        :return: 空域数据
        """
        row = None
        result = None
        h, w = dct_data.shape[0], dct_data.shape[1]
        for i in range(h):
            for j in range(w):
                block = cv2.idct(dct_data[i, j, ...])
                row = block if j == 0 else np.hstack((row, block))
            result = row if i == 0 else np.vstack((result, row))
        return result.astype(np.uint8)

    def dct_extract(self, synthesis, watermark_size):
        """
        从嵌入水印的图像中提取水印
        :param synthesis: 嵌入水印的空域图像
        :param watermark_size: 水印大小
        :return: 提取的空域水印
        """
        w_h, w_w = watermark_size
        recover_watermark = np.zeros(shape=watermark_size)
        synthesis_dct_blocks = self.dct_blkproc(background=synthesis)
        p = np.zeros(self.block_size)
        for h in range(w_h):
            for w in range(w_w):
                for k in range(self.block_size):
                    p[k] = synthesis_dct_blocks[h, w, k, self.block_size - 1]
                if self.corr2(p, self.k1) > self.corr2(p, self.k2):
                    recover_watermark[h, w] = 1
                else:
                    recover_watermark[h, w] = 0
        return recover_watermark


    def mean2(self,x):
        y = np.sum(x) / np.size(x)
        return y


    def corr2(self,a, b):
        """
        相关性判断
        """
        a = a - self.mean2(a)
        b = b - self.mean2(b)
        r = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
        return r



if __name__ == '__main__':
    # 处理输入参数
    parser = argparse.ArgumentParser(description='输入图片和水印，输出加入水印的图片')
    parser.add_argument("-f","--file", help="原图文件路径", type=str, required=True)
    parser.add_argument("-w","--watermark", help="水印文件路径", type=str, required=True)
    parser.add_argument("-o","--output", help="输出路径与文件名", type=str,default=None)
    parser.add_argument("-a", "--alpha",help="尺度控制因子，默认100",default=100, type=int,)
    parser.add_argument("-b", "--blocksize",help="分块大小，默认8",default=8, type=int)
    args = parser.parse_args()
    if not(os.path.exists(args.file)):
        print("原始图片不存在")
        raise ValueError    
    
    if not(os.path.exists(args.watermark)):
        print("水印图片不存在")
        raise ValueError   
    
    output=None
    if args.output==None:
        output=args.file.split(".")[0]+"output.png"
    else:
        output=args.output
    # 超参数设置
    alpha = args.alpha  # 尺度控制因子，控制水印添加强度，决定频域系数被修改的幅度
    blocksize = args.blocksize  # 分块大小
    
    # 2. 初始化DCT算法
    dct_emb = DCT_Embed(backgroundurl=args.file, watermarkurl=args.watermark,block_size=blocksize, alpha=alpha,output=output)
    dct_emb.run()


