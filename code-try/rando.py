
    # if hands:
    #     hand = hands[0]
    #     x, y, w, h = hand['bbox']

    #     imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    #     imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

    #     imgCropShape = imgCrop.shape

    #     aspectRatio = h / w

    #     if aspectRatio > 1:
    #         k = imgSize / h
    #         wCal = math.ceil(k * w)
    #         if imgCrop.size > 0:
    #             ImgResize = cv2.resize(imgCrop, (wCal, imgSize))
    #         else: 
    #             continue
    #         imgResizeShape = imgResize.shape
    #         wGap = math.ceil((imgSize - wCal) / 2)
    #         imgWhite[:, wGap:wCal + wGap] = imgResize


    #     else:
    #         k = imgSize / w
    #         hCal = math.ceil(k * h)
    #         if imgCrop.size >0:
    #             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
    #             imgResizeShape = imgResize.shape
    #             hGap = math.ceil((imgSize - hCal) / 2)
    #             imgWhite[hGap:hCal + hGap, :] = imgResize
    #         else: 
    #             continue
    # 