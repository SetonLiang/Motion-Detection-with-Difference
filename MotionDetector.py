import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import glob
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
mpl.use('TkAgg')


def image_process(directory_path=r'D:\HighwayII\*.png', diff_threshold=30, min_contour_area=100,
                  page_index=210, background_on=True):

    image_files = sorted(glob.glob(directory_path))

    # 创建背景建模器
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    # 创建输出图片对象
    fig = Figure(figsize=(10, 4))

    # 开始图像处理过程
    for k in range(min(0, page_index - 5), page_index + 1):  # 往前回溯5帧（或从第1张开始）来进行背景建模
        x = cv2.imread(image_files[k - 1], cv2.IMREAD_COLOR)
        y = cv2.imread(image_files[k], cv2.IMREAD_COLOR)
        original = y.copy()

        # 背景建模法
        fg_mask = background_subtractor.apply(x)

        # 差分法
        diff = cv2.absdiff(x, y)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary_diff = cv2.threshold(gray_diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # 进行形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_diff = cv2.erode(binary_diff, kernel)
        binary_diff = cv2.dilate(binary_diff, kernel)

        # 合并背景建模法和差分法的结果
        if background_on:
            combined_mask = cv2.bitwise_or(fg_mask, binary_diff)
        else:
            # combined_mask = cv2.bitwise_or(gray_diff, binary_diff)
            combined_mask = binary_diff

        # 寻找轮廓并绘制边界框
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        marked_image = y.copy()
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示结果
        if k in [page_index]:
            # 原图
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')

            # 背景建模法结果
            ax2 = fig.add_subplot(2, 3, 4)
            ax2.imshow(fg_mask, cmap='gray')
            ax2.set_title('Background Subtraction Result')
            ax2.axis('off')

            # 差分法原始结果
            ax3 = fig.add_subplot(2, 3, 2)
            ax3.imshow(gray_diff, cmap='gray')
            ax3.set_title('Difference Result')
            ax3.axis('off')

            # 差分法形态学调整后结果
            ax4 = fig.add_subplot(2, 3, 3)
            ax4.imshow(binary_diff, cmap='gray')
            ax4.set_title('Difference Result (After modified)')
            ax4.axis('off')

            # 两个方法结合之后的结果
            ax5 = fig.add_subplot(2, 3, 5)
            ax5.imshow(combined_mask, cmap='gray')
            ax5.set_title('Combined Result')
            ax5.axis('off')

            # 标记图
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
            ax6.set_title('Marked Image')
            ax6.axis('off')

    return fig


def three_frame_difference(directory_path=r'E:\photo\Images\HighwayII\*.png', diff_threshold=30, min_contour_area=100,
                           page_index=210, background_on=True):

    image_files = sorted(glob.glob(directory_path))

    # 创建背景建模器
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    # 创建输出图片对象
    fig = Figure(figsize=(10, 4))

    # 开始图像处理过程
    for k in range(min(0, page_index - 4), page_index + 1):  # 往前回溯5帧（或从第2张开始）来进行背景建模
        x = cv2.imread(image_files[k - 2], cv2.IMREAD_COLOR)
        y = cv2.imread(image_files[k - 1], cv2.IMREAD_COLOR)
        z = cv2.imread(image_files[k], cv2.IMREAD_COLOR)
        original = z.copy()

        # 背景建模法
        fg_mask = background_subtractor.apply(z)

        # 三帧差分法
        diff1 = cv2.absdiff(x, y)
        diff2 = cv2.absdiff(y, z)
        gray_diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
        gray_diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.bitwise_or(gray_diff1, gray_diff2)

        _, binary_diff1 = cv2.threshold(gray_diff1, diff_threshold, 255, cv2.THRESH_BINARY)
        _, binary_diff2 = cv2.threshold(gray_diff2, diff_threshold, 255, cv2.THRESH_BINARY)

        binary_diff = cv2.bitwise_or(binary_diff1, binary_diff2)

        # 进行形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_diff = cv2.erode(binary_diff, kernel)
        binary_diff = cv2.dilate(binary_diff, kernel)

        # 合并背景建模法和差分法的结果
        if background_on:
            combined_mask = cv2.bitwise_or(fg_mask, binary_diff)
        else:
            combined_mask = binary_diff

        # 寻找轮廓并绘制边界框
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        marked_image = z.copy()
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示结果
        if k in [page_index]:
            # 原图
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')

            # 背景建模法结果
            ax2 = fig.add_subplot(2, 3, 4)
            ax2.imshow(fg_mask, cmap='gray')
            ax2.set_title('Background Subtraction Result')
            ax2.axis('off')

            # 三帧差分法结果
            ax3 = fig.add_subplot(2, 3, 2)
            ax3.imshow(gray_diff, cmap='gray')
            ax3.set_title('Three Frame Difference Result')
            ax3.axis('off')

            # 形态学操作结果
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.imshow(binary_diff, cmap='gray')
            ax3.set_title('Difference Result (After modified)')
            ax3.axis('off')

            # 两个方法结合之后的结果
            ax4 = fig.add_subplot(2, 3, 5)
            ax4.imshow(combined_mask, cmap='gray')
            ax4.set_title('Combined Result')
            ax4.axis('off')

            # 标记图
            ax5 = fig.add_subplot(2, 3, 6)
            ax5.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
            ax5.set_title('Marked Image')
            ax5.axis('off')

    return fig


directory = ""  # 全局变量，保存当前选择的目录路径
current_page = 6  # 全局变量，保存当前页码
threshold = 30  # 全局变量，保存阈值，默认为30
max_page = 6  # 全局变量，最大页数
background_apply = True  # 全局变量，是否启用背景法
three_process_apply = False


def browse_directory():
    global directory
    directory = filedialog.askdirectory(initialdir="Desktop", title="选择文件夹")
    if directory:
        text_entry.delete(0, tk.END)
        text_entry.insert(tk.END, directory)
        page_number.delete(0, tk.END)
        page_number.insert(tk.END, '6')
        threshold_entry.delete(0, tk.END)
        threshold_entry.insert(tk.END, '30')
        access_directory()


def switch_process():
    global three_process_apply
    three_process_apply = not three_process_apply
    if three_process_apply:
        switch_button['text'] = "三帧法已开启"
        switch_button['bg'] = 'lightgreen'
    else:
        switch_button['text'] = "三帧法已关闭"
        switch_button['bg'] = 'pink'
    access_directory()


def access_directory():
    global directory, current_page, max_page, threshold, background_apply, three_process_apply
    directory = text_entry.get()
    image_files = sorted(glob.glob(directory + r'/*.png'))
    max_page = len(image_files) - 1
    threshold_entry.delete(0, tk.END)
    threshold_entry.insert(tk.END, str(threshold))
    if directory:
        try:
            dir_path = directory + r'/*.png'
            if three_process_apply:
                plot = three_frame_difference(directory_path=dir_path, diff_threshold=threshold,
                                     page_index=current_page, background_on=background_apply)
            else:
                plot = image_process(directory_path=dir_path, diff_threshold=threshold,
                                 page_index=current_page, background_on=background_apply)
            show_image(plot)
        except Exception as e:
            messagebox.showerror("访问目录", f"访问目录时出错：{str(e)}")
    else:
        messagebox.showwarning("访问目录", "请输入一个有效的目录路径")


def show_image(image):
    fig = image
    clear_frame(big_image_frame)
    canvas = FigureCanvasTkAgg(fig, big_image_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)


def clear_frame(frame):
    for widgets in frame.winfo_children():
        widgets.destroy()


def previous_page():
    global current_page
    if current_page > 1:
        current_page -= 1
        page_number.delete(0, tk.END)
        page_number.insert(tk.END, str(current_page))
        access_directory()
    else:
        messagebox.showwarning("翻页", "已经在第一页")


def next_page():
    global current_page, max_page
    if current_page < max_page:
        current_page += 1
        page_number.delete(0, tk.END)
        page_number.insert(tk.END, str(current_page))
        access_directory()
    else:
        messagebox.showwarning("翻页", "已经在最后一页")


def apply_threshold():
    global threshold
    try:
        new_threshold = int(threshold_entry.get())
        threshold = new_threshold
        threshold_entry.delete(0, tk.END)
        threshold_entry.insert(tk.END, str(threshold))
        messagebox.showinfo("应用阈值", f"阈值已成功更改为 {threshold}")
        access_directory()
    except ValueError:
        messagebox.showwarning("应用阈值", "请输入有效的正整数作为阈值")


def update_page(_=''):
    global current_page, max_page
    if 1 <= int(page_number.get()) <= max_page:
        current_page = int(page_number.get())
        page_number.delete(0, tk.END)
        page_number.insert(tk.END, str(current_page))
        access_directory()
    else:
        messagebox.showwarning("翻页", "页码不存在")


def apply_background():
    global background_apply
    background_apply = not background_apply
    if background_apply:
        background_button['text'] = "背景法已开启"
        background_button['bg'] = 'lightgreen'
    else:
        background_button['text'] = "背景法已关闭"
        background_button['bg'] = 'pink'
    access_directory()


# 创建主窗口
window = tk.Tk()
window.title("运动检测")

# 创建顶部文本框和按钮
top_frame = tk.Frame(window)
top_frame.pack(side=tk.TOP, pady=10)

text_entry = tk.Entry(top_frame)
text_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
text_entry.config(width=40)  # 将宽度翻倍

browse_button = tk.Button(top_frame, text="浏览", command=browse_directory)
browse_button.pack(side=tk.LEFT, padx=10)

# access_button = tk.Button(top_frame, text="访问", command=access_directory)
# access_button.pack(side=tk.LEFT)


# 添加阈值输入框和应用阈值按钮
threshold_frame = tk.Frame(window)
threshold_frame.pack(side=tk.TOP, pady=10)

threshold_label = tk.Label(threshold_frame, text="阈值：")
threshold_label.pack(side=tk.LEFT)

threshold_entry = tk.Entry(threshold_frame)
threshold_entry.pack(side=tk.LEFT, padx=5)

apply_button = tk.Button(threshold_frame, text="应用阈值", command=apply_threshold)
apply_button.pack(side=tk.LEFT)

gap_label = tk.Label(threshold_frame, text=" ")
gap_label.pack(side=tk.LEFT)

background_button = tk.Button(threshold_frame, text="背景法已开启", bg='lightgreen', command=apply_background)
background_button.pack(side=tk.LEFT)

gap_label = tk.Label(threshold_frame, text=" ")
gap_label.pack(side=tk.LEFT)

switch_button = tk.Button(threshold_frame, text="三帧法已关闭", bg='pink', command=switch_process)
switch_button.pack(side=tk.LEFT)

# 创建图片窗口的父容器
image_container = tk.Frame(window)
image_container.pack(side=tk.TOP, pady=10)

# 创建大的图片窗口
big_image_frame = tk.Frame(image_container, width=600, height=400, relief=tk.SOLID, borderwidth=1)
big_image_frame.pack(padx=10, pady=10)

# 创建底部按钮和页码文本框
bottom_frame = tk.Frame(window)
bottom_frame.pack(side=tk.BOTTOM, pady=10)

previous_button = tk.Button(bottom_frame, text="上一页", command=previous_page)
previous_button.pack(side=tk.LEFT)

page_number = tk.Entry(bottom_frame)
page_number.bind('<Return>', update_page)
page_number.pack(side=tk.LEFT, padx=5)
page_number.config(width=5)  # 将宽度减半

next_button = tk.Button(bottom_frame, text="下一页", command=next_page)
next_button.pack(side=tk.LEFT)


# 运行主循环
window.mainloop()
