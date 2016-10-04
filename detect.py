import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import sys
# import peakutils
from libs import detect_peaks

def process_img(filename):
  img = cv2.imread('batemans/' + filename, 1)
  new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  height, width, depth = img.shape

  canvas = np.zeros((height, width), np.uint8)

  print('Computing...')

  def calcEntropy(img):
    #hist,_ = np.histogram(img, np.arange(0, 256), normed=True)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.ravel()/hist.sum()
    #logs = np.nan_to_num(np.log2(hist))
    logs = np.log2(hist+0.00001)
    #hist_loghist = hist * logs
    entropy = -1 * (hist*logs).sum()
    return entropy

  for i in range(0, height):
    for j in range(0, width):
      if img[i,j][2] >= img[i,j][1] and img[i,j][2] >= img[i,j][0]:
        canvas[i,j] = img[i,j][2]
      else:
        canvas[i,j] = 0

  cv2.imwrite('dumps/'+filename[:-4]+'_01_maxthresh.png', canvas)

  # radius = 60

  # # kernel = np.ones(blur, np.float32) / val
  # kernel = np.zeros((radius * 2, radius * 2), np.float32)
  # kernel -= 0.0003
  # cv2.circle(kernel, (radius, radius), radius, 0.0003, -1)
  # canvas = cv2.filter2D(canvas, -1, kernel)
  # # canvas = cv2.equalizeHist(canvas)
  # # cv2.imwrite('kernel.png',kernel)
  # # canvas = cv2.normalize(canvas)
  # # print(canvas[233][1063])


  # # print(kernel)


  # # new_canvas = np.zeros((height, width), np.uint8)
  # maxval = canvas.max()
  # ret, mask = cv2.threshold(canvas, maxval * 0.5, 255, cv2.THRESH_BINARY)

  # erode_kernel = np.ones((5,5),np.uint8)


  # new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # canvas = cv2.bitwise_and(new_img, mask)

  # canvas = cv2.adaptiveThreshold(canvas, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,2)
  # # canvas = cv2.normalize(canvas)
  # # canvas = cv2.Canny(canvas,0,100,3)

  # canvas = cv2.bitwise_not(canvas)

  # mask = cv2.erode(mask, erode_kernel)
  # canvas = cv2.bitwise_and(canvas, mask)

  # kernel = np.ones((2,2),np.uint8)
  # canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)




  # cv2.imwrite('test.png', canvas)

  height, width, = canvas.shape
  subwin_size = 40
  total = (height-subwin_size - subwin_size) * (width-subwin_size - subwin_size)
  print("total = " + str(total))
  count = 0

  mask = np.zeros((subwin_size * 2, subwin_size * 2), np.uint8)
  cv2.circle(mask, (subwin_size, subwin_size), subwin_size, 255, -1)

  # cv2.imwrite('dumps/'+filename[:-4]+'_02_mask.png', mask)



  radius = 30
  kernel = np.zeros((radius * 2, radius * 2), np.float32)
  cv2.circle(kernel, (radius, radius), radius, 3.0/(0.78 * math.pow(2*radius, 2)), -1)
  mask = cv2.filter2D(canvas, -1, kernel)


  ret, mask = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)

  cv2.imwrite('dumps/'+filename[:-4]+'_03_masked.png', mask)


  radius = 50
  kernel = np.zeros((radius * 2, radius * 2), np.uint8)
  cv2.circle(kernel, (radius, radius), radius, 255, -1)
  mask = cv2.dilate(mask, kernel)
  # canvas = cv2.bitwise_and(canvas, mask)

  for i in range(0, height):
    for j in range(0, width):
      if mask[i,j] == 0:
        new_img[i, j] = 0
        # canvas[i, j] = 0

  cv2.imwrite('dumps/'+filename[:-4]+'_04_masked.png', new_img)

  new_img = cv2.Canny(new_img,70,0,0)

  cv2.imwrite('dumps/'+filename[:-4]+'_05_cannyd.png', new_img)
  # for a in range(0,255,10):
  #   for b in range(0,200, 20):
  #     if b < 100:
  #       if a < 100:
  #         cv2.imwrite("test_canny_a-0{a}_b-0{b}.png".format(a=a, b=b), )
  #       else:
  #         cv2.imwrite("test_canny_a-{a}_b-0{b}.png".format(a=a, b=b), cv2.Canny(new_img,a,b,0))
  #     else:
  #       if a < 100:
  #         cv2.imwrite("test_canny_a-0{a}_b-{b}.png".format(a=a, b=b), cv2.Canny(new_img,a,b,0))
  #       else:
  #         cv2.imwrite("test_canny_a-{a}_b-{b}.png".format(a=a, b=b), cv2.Canny(new_img,a,b,0))
  # new_img = cv2.Canny(new_img,50,100,3)

  radius = 10
  kernel = np.zeros((radius * 2, radius * 2), np.uint8)
  cv2.circle(kernel, (radius, radius), radius, 255, -1)
  mask = cv2.erode(mask, kernel)
  new_img = cv2.bitwise_and(new_img, mask)

  cv2.imwrite('dumps/'+filename[:-4]+'_06_cannyd.png', new_img)

  #########
  #   Remove low count regions
  #########

  radius = 8
  subwin_size = radius
  thresh = radius
  mask = np.zeros((radius * 2, radius * 2), np.uint8)
  cv2.circle(mask, (radius, radius), radius, 255, -1)
  for y in xrange(subwin_size, height-subwin_size):
    for x in xrange(subwin_size, width-subwin_size):
      subwin = canvas[y-subwin_size:y+subwin_size, x-subwin_size:x+subwin_size].copy()
      subwin = cv2.bitwise_and(subwin, mask)
      total_count = cv2.countNonZero(subwin)
      if total_count < 80:
        new_img[y][x] = 0

  cv2.imwrite('dumps/'+filename[:-4]+'_07_lowcount_removed_stage1.png', new_img)

  outs = []
  radius = 15
  subwin_size = radius
  thresh = radius
  mask = np.zeros((radius * 2, radius * 2), np.uint8)
  cv2.circle(mask, (radius, radius), radius, 255, -1)
  for y in xrange(subwin_size, height-subwin_size):
    for x in xrange(subwin_size, width-subwin_size):
      subwin = new_img[y-subwin_size:y+subwin_size, x-subwin_size:x+subwin_size].copy()
      subwin = cv2.bitwise_and(subwin, mask)
      total_count = cv2.countNonZero(subwin)
      if total_count > 0:
        outs.append(total_count)
      if total_count < 100:
        new_img[y][x] = 0


  cv2.imwrite('dumps/'+filename[:-4]+'_08_lowcount_removed_stage2.png', new_img)

  def fit_best(loc, new_img):
    height, width, = new_img.shape
    max_rad = 60
    if loc[0] < max_rad:
      max_rad = loc[0]
    if loc[1] < max_rad:
      max_rad = loc[1]
    if loc[0] > (width - max_rad):
      max_rad = width - loc[0]
    if loc[1] > (height - max_rad):
      max_rad = height - loc[1]

    x = loc[0]
    y = loc[1]
    diffs = []
    last = 0
    radius = 5
    for rad in range(5, max_rad, 5):
      kernel = np.zeros((max_rad * 2, max_rad * 2), np.uint8)
      cv2.circle(kernel, (max_rad, max_rad), rad, 255, -1)
      frame = new_img[y-max_rad:y+max_rad, x-max_rad:x+max_rad].copy()
      subwin = cv2.bitwise_and(frame, kernel)
      # cv2.imwrite('test_'+str(rad)+'.png', subwin)
      count = cv2.countNonZero(subwin)
      diff = count - last
      if len(diffs) > 0:
        if diff < diffs[-1]:
          break
        else:
          radius = rad
      else:
        radius = rad
      last = count
      diffs.append(diff)
    return radius

  cv2.rectangle(new_img, (3, 3), (width-3, height-3), 0, 10)
  main_img = new_img.copy()

  def get_templated(img_in):
    radius = 50
    kernel = np.zeros((radius * 2, radius * 2), np.float32)
    for i, mag in zip([40, 10], [50, 30]):
      cv2.circle(kernel, (radius, radius), i, mag, -1)
    # cv2.imwrite('kernel.png', kernel)
    return cv2.filter2D(np.float32(img_in.copy()), -1, kernel)

  new_img = get_templated(main_img)

  cv2.imwrite('dumps/'+filename[:-4]+'_09_readyforblanking.png', main_img)

  # cv2.imwrite('test.png', new_img)

  output_locations = []
  output_radaii = []
  blank_padding = 30
  minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(new_img)
  count = 0
  # cv2.imwrite('test_00.png', main_img)
  height, width, = new_img.shape
  while maxVal > 500000 and count < 30:
    if maxLoc[0] < 30 or maxLoc[1] < 30 or maxLoc[0] > width-30 or maxLoc[1] > height-30:
      # print('trying ', maxLoc, radius)
      count += 1
      cv2.circle(main_img, maxLoc, 30, 0, -1)
      cv2.circle(new_img, maxLoc, 30, 0, -1)
      minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(new_img)
      continue
    radius = fit_best(maxLoc, main_img)
    if radius > 15:
      output_locations.append(maxLoc)
      output_radaii.append(radius)
      print(maxLoc, radius)
    cv2.circle(main_img, maxLoc, radius+blank_padding, 0, -1)
    cv2.circle(new_img, maxLoc, radius+blank_padding, 0, -1)
    if radius > 15:
      if count < 10:
        cv2.imwrite('dumps/'+filename[:-4]+'_10_0'+str(count)+'_blanking_rad-'+str(radius)+'_'+str(maxVal)+'.png', main_img)
      else:
        cv2.imwrite('dumps/'+filename[:-4]+'_10_'+str(count)+'_blanking_rad-'+str(radius)+'_'+str(maxVal)+'.png', main_img)
    count += 1
    new_img = get_templated(main_img)
    # new_img = cv2.filter2D(np.float32(main_img.copy()), -1, kernel)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(new_img)


  print(output_locations)
  print(output_radaii)

  for i, (loc, rad) in enumerate(zip(output_locations, output_radaii)):
    x0 = loc[0] - rad
    x1 = loc[0] + rad
    y0 = loc[1] - rad
    y1 = loc[1] + rad
    berg = img[y0:y1,x0:x1,:]
    cv2.imwrite('dumps/'+filename[:-4]+'_11_'+str(i)+'_region_rad-'+str(rad)+'_l0-'+str(loc[0])+'_l1-'+str(loc[1])+'.png', berg)
    cv2.rectangle(img, (loc[0]-rad, loc[1]-rad), (loc[0]+rad, loc[1]+rad), (0, 0, 255), 5)

  print('finished' , filename)
  cv2.imwrite('dumps/'+filename[:-4]+'_11_result.png', img)


filenames =os.listdir('batemans')
filenames.sort()
for filename in filenames:
# for filename in ['Left Camera Image 1067.bmp']:
  process_img(filename)
  # break
# canvas_out = np.zeros((height, width), np.float32)

# counts = []
# locs = []
# print("Scanning image for count thresholds")
# for y in xrange(subwin_size, height-subwin_size):
#   for x in xrange(subwin_size, width-subwin_size):
#     subwin = canvas[y-subwin_size:y+subwin_size, x-subwin_size:x+subwin_size].copy()
#     subwin = cv2.bitwise_and(subwin, mask)
#     total_count = cv2.countNonZero(subwin)
#     if total_count > 100:
#       counts.append(total_count)
#       locs.append((y, x))
#       canvas_out[y][x] = total_count

# cv2.imwrite('test.png', canvas)
# cv2.imwrite('test_post.png', canvas_out)

# print("Done. Finding peaks...")
# # with open('counts.csv', 'w') as f:
# #   for x in counts:
# #     f.write(str(x)+'\n')


# indexes = list(detect_peaks.detect_peaks(counts, mph=1000, mpd=40000))
# print(indexes)
# canvas_horz = np.zeros((height, width), np.uint8)

# count = 0
# for y in xrange(subwin_size, height-subwin_size):
#   for x in xrange(subwin_size, width-subwin_size):
#     if canvas_out[y][x] > 0:
#       if count in indexes:
#         canvas_horz[y][x] = 255
#       else:
#         canvas_horz[y][x] = 0
#       count += 1
#     else:
#       canvas_horz[y][x] = 0








# height, width, = canvas.shape
# subwin_size = 40
# total = (height-subwin_size - subwin_size) * (width-subwin_size - subwin_size)
# print("total = " + str(total))
# count = 0

# mask = np.zeros((subwin_size * 2, subwin_size * 2), np.uint8)
# cv2.circle(mask, (subwin_size, subwin_size), subwin_size, 255, -1)
# cv2.imwrite('kernel_preview.png', mask)

# canvas_out = np.zeros((height, width), np.uint32)

# counts = []
# locs = []
# print("Scanning image for count thresholds")
# for x in xrange(subwin_size, width-subwin_size):
#   for y in xrange(subwin_size, height-subwin_size):
#     subwin = canvas[y-subwin_size:y+subwin_size, x-subwin_size:x+subwin_size].copy()
#     subwin = cv2.bitwise_and(subwin, mask)
#     total_count = cv2.countNonZero(subwin)
#     if total_count > 0:
#       counts.append(total_count)
#       locs.append((y, x))
#       canvas_out[y][x] = total_count

# print("Done. Finding peaks...")
# # with open('counts.csv', 'w') as f:
# #   for x in counts:
# #     f.write(str(x)+'\n')


# indexes = list(detect_peaks.detect_peaks(counts, mph=1000, mpd=40000))
# print(indexes)
# canvas_vert = np.zeros((height, width), np.uint8)

# count = 0
# for x in xrange(subwin_size, width-subwin_size):
#   for y in xrange(subwin_size, height-subwin_size):
#     if canvas_out[y][x] > 0:
#       if count in indexes:
#         canvas_vert[y][x] = 255
#       else:
#         canvas_vert[y][x] = 0
#       count += 1
#     else:
#       canvas_vert[y][x] = 0

# # cv2.imwrite('test_vert.png', canvas_vert)
# # cv2.imwrite('test_horz.png', canvas_horz)

# canvas = np.zeros((height, width), np.uint8)
# canvas = cv2.bitwise_or(canvas_vert, canvas_horz)

# cv2.imwrite('test_vert.png', canvas)

# counts = list(np.array(counts) / sum(counts))
# indexes = peakutils.peak.indexes(np.array(counts), thres=1.0/max(counts), min_dist=subwin_size)
# print(indexes)
# print(len(indexes))
# plot_peaks(np.array(counts), indexes, mph=7, mpd=2)
# print("Done. plotting counts...")
# # plt.plot(counts)
# print('Peaks found count = ' + str(len(indexes)))
# indicies = list(indexes)
# print("Done, plotting peaks")
# xs = []
# ys = []
# for xpos, i in enumerate(counts):
#   if i in indicies:
#     xs.append(xpos)
#     ys.append(3750)

# print(xs)
# print(ys)

# plt.scatter(xs,ys, color='red')
# print("Done, showing plots")
# plt.show()


# canvas = np.zeros((height, width), np.uint8)

# for y in xrange(subwin_size, height-subwin_size):
#   for x in xrange(subwin_size, width-subwin_size):
#     px = canvas_out[y,x]
#     if px in indexes:
#       cv2.circle(canvas, (x, y), distance/2, 255, -1)
    # subwin = cv2.bitwise_and(subwin, mask)
    # total_count = cv2.countNonZero(subwin)
    # if total_count > 0:
    #   counts.append(total_count)
    #   locs.append((y, x))
    # if total_count > 1000:
    #   canvas_out[y][x] = total_count

# for ind in indexes:
#   i = indexes.index(ind)
#   print(locs[i])

# print(len(indexes))







# h, w, = canvas.shape
# subwin_size = 30
# # edge_margin = 2
# total = (h-subwin_size - subwin_size) * (w-subwin_size - subwin_size)
# print("total = " + str(total))
# count = 0

# kernel = np.zeros((subwin_size * 2, subwin_size * 2), np.uint8)
# cv2.rectangle(kernel, (edge_margin-1, edge_margin-1), (2*subwin_size-edge_margin,2*subwin_size-edge_margin), (255), -1)
# cv2.imwrite('preview.png', kernel)

# height, width, depth = img.shape
# canvas_out = np.zeros((height, width), np.uint8)

# from scipy.special import entr
# maxz = 0
# for y in xrange(subwin_size, h-subwin_size):
#   for x in xrange(subwin_size, w-subwin_size):
#     subwin = canvas[y-subwin_size:y+subwin_size, x-subwin_size:x+subwin_size].copy()
#     total_count = cv2.countNonZero(subwin)
#     if total_count > 0:
#       tr_count = 0
#       last = 0
#       for i in xrange(0, 2*subwin_size):
#         for j in xrange(0, 2*subwin_size):
#           if subwin[i][j] != last:
#             tr_count += 1
#             last = subwin[i][j]

#       for i in xrange(0, 2*subwin_size):
#         for j in xrange(0, 2*subwin_size):
#           if subwin[j][i] != last:
#             tr_count += 1
#             last = subwin[j][i]


#       tr_count = float(tr_count) / float(total_count)

#       # if tr_count > maxz:
#       #   maxz = tr_count
#       if tr_count > 1.5:
#         print(tr_count)

#       if tr_count > 1.5:
#         canvas_out[y][x] = 255

#     else:
#       canvas_out[y][x] = 0
#     count += 1

# print(maxz)
# print(minz)
    # if total_count > 0:
    #   if count < 270000:
    #     cv2.imwrite("dump/{num}_pre.png".format(num=count), subwin)
    #   # if count == 452194:
    #   #   cv2.imwrite('preview2_pre.png', subwin)
    #   #   cv2.imwrite('preview2_kern.png', kernel)

    #   subwin = cv2.bitwise_and(subwin, kernel)

    #   if count < 270000:
    #     cv2.imwrite("dump/{num}_prez.png".format(num=count), subwin)

    #   # if count == 452194:
    #   #   cv2.imwrite('preview2_post.png', subwin)

    #   border_count = cv2.countNonZero(subwin)
    #   if float(border_count)/total_count > 0.8:
    #     canvas_out[y][x] = total_count
    #     print(count, total_count, border_count, float(border_count)/total_count)
    #   else:
    #     canvas[y][x] = 0
      # entropy = calcEntropy(subwin)    # Calculate entropy
      # canvas_out[y][x] = entropy * 100.0
    # if x == subwin_size:
    #   print("{pc}%".format(pc=round(float(count)/total * 100.0, 2)))
# canvas_out = canvas
# radius = 20
# kernel = np.zeros((radius * 2, radius * 2), np.float32)
# # kernel -= 15/(0.78 * math.pow(2*radius, 2))
# cv2.circle(kernel, (radius, radius), radius, 3.0/(0.78 * math.pow(2*radius, 2)), -1)
# canvas = cv2.filter2D(canvas, -1, kernel)


# kernel = np.ones((2,2),np.uint8)
# mask = cv2.erode(canvas, kernel)
# canvas = cv2.bitwise_and(canvas, mask)


# kernel = np.ones((10,10),np.uint8)
# mask = cv2.erode(mask, kernel)
# canvas = cv2.bitwise_and(canvas, mask)
# canvas = cv2.equalizeHist(canvas)
# canvas = cv2.filter2D(canvas, -1, kernel)
# cv2.imshow('image', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# radius = 42
# kernel = np.zeros((radius * 2, radius * 2), np.float32)
# cv2.circle(kernel, (radius, radius), radius, 1.0/(radius * radius), -1)
# canvas = cv2.filter2D(canvas, -1, kernel)


