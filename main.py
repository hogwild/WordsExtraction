import cv2
import numpy as np


def near_wall_test(floor, wall, max_dist, delta_x, delta_y):
    '''判定floor往一个方向走时是否会在max_dist步内遇到wall'''

    if (delta_x * delta_y != 0):
        raise Exception("Unsupported")
    result = np.zeros_like(wall, dtype='int8')
    range_x = range(delta_x, wall.shape[0])
    if (delta_x < 0):
        range_x = range(wall.shape[0] + delta_x - 1, -1, -1)
    range_y = range(delta_y, wall.shape[1])
    if (delta_y < 0):
        range_y = range(wall.shape[1] + delta_y - 1, -1, -1)
    result[wall] = max_dist
    if (delta_x == 0):
        for y in range_y:
            result[:, y] = np.maximum(result[:, y], floor[:, y] * result[:, y - delta_y] - 1)
    else:
        for x in range_x:
            result[x, :] = np.maximum(result[x, :], floor[x, :] * result[x - delta_x, :] - 1)
    return np.logical_and(result > 0, floor)


current_emit = 0


def emit(id):
    global current_emit, fps, cap
    if (current_emit != id):
        current_emit = id
        print('%.6f\t%d' % (cap.get(cv2.CAP_PROP_POS_FRAMES) / fps, id))


def get_current_emit():
    global current_emit
    return current_emit


def claim_rect(id, condition, output, x, y, h, w):
    global EMIT_THERSHOLD
    if (condition[x:x + h, y:y + w].sum() > EMIT_THERSHOLD):
        emit(id)
    if (get_current_emit() == id):
        color = [0, 255, 255]
    else:
        color = [255, 0, 255]
    cv2.rectangle(output, (y, x), (y + w - 1, x + h - 1), color)


if __name__ == '__main__':
    global fps, cap, EMIT_THERSHOLD

    song_id = 1
    if (song_id == 1):
        cap = cv2.VideoCapture(R'I:\Dataset\avi\何洁-请不要对我说Sorry(MTV)-国语-流行.avi')
        block_height = 53
        block_width = 47
        line1_x = 1
        line1_y = 64
        line2_x = 61
        line2_y = -66
        max_text_length = 8
        START_FRAME = 1400
        EMIT_THERSHOLD = 4
    elif (song_id == 2):
        cap = cv2.VideoCapture(R'I:\Dataset\avi\毛宁-晚秋(MTV)-国语-流行.avi')
        block_height = 53
        block_width = 44
        line1_x = 14
        line1_y = 81
        line2_x = 71
        line2_y = -66
        max_text_length = 13
        START_FRAME = 400
        EMIT_THERSHOLD = 2
    elif (song_id == 3):
        cap = cv2.VideoCapture(R'I:\Dataset\avi\蒋大为-敢问路在何方(演唱会)-国语-影视插曲.avi')
        block_height = 53
        block_width = 44
        line1_x = 14
        line1_y = 81
        line2_x = 71
        line2_y = -66
        max_text_length = 13
        START_FRAME = 5350
        EMIT_THERSHOLD = 2
    else:
        raise NotImplementedError()
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 限定字幕可能出现的范围：底部1/3，可选参数
    X = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 3
    Y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(1, START_FRAME)
    countmap = np.zeros((X, Y), dtype='int8')
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (not ret):
            break
        frame = frame[-X:, :]
        output = np.zeros_like(frame)
        # 输入是bgr，先转rgb
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # 判定白底部分（潜在未唱文字前景）
        white_area = np.logical_and(hsv[:, :, 1] < 10, hsv[:, :, 2] > 170)
        # 判定黑色部分（潜在未唱文字边框）
        black_area = hsv[:, :, 2] < 70
        # 简单扩展一下黑色的范围，填补黑-白过渡区域
        black_area = cv2.distanceTransform(1 - black_area.astype('uint8'), maskSize=3, distanceType=cv2.DIST_C) < 2
        # 判定蓝色部分（潜在已唱文字前景，目前颜色指定为蓝色）
        blue_area = np.logical_and(np.logical_and(hsv[:, :, 0] <= 125, hsv[:, :, 0] >= 115),
                                   np.logical_and(hsv[:, :, 1] >= 210, hsv[:, :, 2] >= 20))

        # 三方向判断白底周围是否是黑色部分，还有一个方向(-y方向)不做判断
        # 因为在跑马字滚动到一半的时候，白底文字往左走可能会遇到蓝色而非黑色
        condition1 = near_wall_test(white_area, black_area, 75, 1, 0)
        condition2 = near_wall_test(white_area, black_area, 75, -1, 0)
        condition3 = near_wall_test(white_area, black_area, 75, 0, -1)
        condition = np.logical_and(np.logical_and(condition1, condition2), condition3)

        # 如果该像素在过去频繁被判定condition==True且突然变成blue_area
        # 那么它很有可能是字幕变化部分
        # countmap记录该像素近期判定为condition==True的频繁度
        switch_cond = np.logical_and(countmap > 15, blue_area)
        countmap -= 1
        countmap = np.maximum(0, countmap)
        countmap[condition] += 2
        countmap = np.minimum(30, countmap)
        countmap[blue_area] = 0

        output[condition] = [255, 0, 0]
        output[blue_area] = [0, 0, 255]
        output[switch_cond] = [0, 255, 255]

        for i in range(max_text_length):
            pos_y = line1_y + block_width * i
            claim_rect(i + 1, switch_cond, output, line1_x, pos_y, block_height, block_width)

        for i in range(max_text_length):
            pos_y = Y + line2_y - block_width * (i + 1)
            claim_rect(-(i + 1), switch_cond, output, line2_x, pos_y, block_height, block_width)

        cv2.imshow('frame', frame)
        cv2.imshow('output', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
