import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("input.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

b, g, r = cv2.split(img)

hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])


equ_b = cv2.equalizeHist(b)
equ_g = cv2.equalizeHist(g)
equ_r = cv2.equalizeHist(r)
equ_gray = cv2.equalizeHist(img_gray)

equ = cv2.merge((equ_b, equ_g, equ_r))

hist_equ_b = cv2.calcHist([equ_b], [0], None, [256], [0, 256])
hist_equ_g = cv2.calcHist([equ_g], [0], None, [256], [0, 256])
hist_equ_r = cv2.calcHist([equ_r], [0], None, [256], [0, 256])
hist_equ_gray = cv2.calcHist([equ_gray], [0], None, [256], [0, 256])

plt.figure(figsize=(15, 15))

plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Orjinal Görüntü')

plt.subplot(3, 4, 2)
plt.plot(hist_b, color='b')
plt.title('mavi histogram')

plt.subplot(3, 4, 3)
plt.plot(hist_g, color='g')
plt.title('yeşil histogram')

plt.subplot(3, 4, 4)
plt.plot(hist_r, color='r')
plt.title('kırmızı histogram')

plt.subplot(3, 4, 5)
plt.imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
plt.title('eşitlenmiş Görüntü')

plt.subplot(3, 4, 6)
plt.plot(hist_equ_b, color='b')
plt.title('eşitlenmiş mavi histogram')

plt.subplot(3, 4, 7)
plt.plot(hist_equ_g, color='g')
plt.title('eşitlenmiş yeşil histogram')

plt.subplot(3, 4, 8)
plt.plot(hist_equ_r, color='r')
plt.title('eşitlenmiş kırmızı histogram')

plt.subplot(3, 4, 9)
plt.imshow(cv2.cvtColor(equ_gray, cv2.COLOR_GRAY2RGB))
plt.title('eşitlenmiş gri Görüntü')

plt.subplot(3, 4, 10)
plt.plot(hist_equ_gray, color='r')
plt.title('eşitlenmiş gri histogram')


plt.tight_layout()
plt.show()





