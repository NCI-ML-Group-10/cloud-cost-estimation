FROM nginx:alpine
EXPOSE 80
# 拷贝前端构建产物
COPY build/ /usr/share/nginx/html

# 覆盖nginx配置
COPY nginx.conf /etc/nginx/conf.d/default.conf
