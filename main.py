from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

#将图像显示以及销毁整合为函数cv_show（）
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


#读取图像和预处理cv2.imread（）
image = cv2.imread("img.png")

#处理为灰度图cv2.cvtColor（）并且进行高斯滤波操作cv2.GaussianBlur（）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)

#边缘处理图像cv2.Canny（）,并且进行闭操作对图像边缘封闭cv2.morphologyEx（）
edged = cv2.Canny(blur, 50, 100)
kernel = np.ones((3,3),dtype=np.uint8)
edged = cv2.morphologyEx(edged,cv2.MORPH_CLOSE,kernel)

#寻找轮廓cv2.findContours（）
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#对轮廓点进行从左到右排序contours.sort_contours（）
(cnts, _) = contours.sort_contours(cnts)

#过滤操作，去除面积不够大的轮廓cv2.contourArea()，将其认为噪音点丢弃
cnts = [x for x in cnts if cv2.contourArea(x) > 100]


#设定参考尺寸，并且将其设定为2cmX2cm cv2.minAreaRect（）cv2.boxPoints（）
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2
pixel_per_cm = dist_in_pixel/dist_in_cm

#绘制剩余的轮廓cv2.drawContours（）
for cnt in cnts:
	box = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)

	#计算各中点
	#计算中点集之间的距离
	(tl, tr, br, bl) = box
	cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
	mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
	mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))

	#计算中点集之间的距离euclidean（）
	#并且通过将相应的欧几里德距离除以 ret 值来计算物体尺寸
	wid = euclidean(tl, tr)/pixel_per_cm
	ht = euclidean(tr, br)/pixel_per_cm
	cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv_show("",image)