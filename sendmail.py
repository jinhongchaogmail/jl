import os
import smtplib
from email.message import EmailMessage
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# 创建邮件对象
msg = MIMEMultipart()

# 设置邮件内容
msg['Subject'] = 'Hello from Python' # 邮件主题
msg['From'] = 'jinhongchao@qq.com' # 发件人邮箱
msg['To'] = '7007jhc@gmail.com' # 收件人邮箱
msg.preamble = 'The task is done, please review.' # 邮件正文

directory = "."
for filename in os.listdir(directory):
    # 检查文件是否是图片
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 打开图片文件
        with open(os.path.join(directory, filename), 'rb') as f:
            # 创建一个 MIMEImage 对象
            img = MIMEImage(f.read())
            # 添加图片到邮件中
            msg.attach(img)

# 连接到 Gmail 服务器
server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

# 登录你的 Gmail 账号
server.login('7007jhc@gmail.com', 'icqe msgl ucfc zype')

# 发送邮件
server.send_message(msg)

# 关闭连接
server.quit()
