# --- Stage 1: Build ---
FROM node:22-alpine as builder

WORKDIR /app

# Dependencies
COPY web_app/frontend/package.json web_app/frontend/package-lock.json ./
# Avoid npm peer dependency resolution failures inside the build environment
# Use legacy peer deps to match the host dev setup and reduce build-time conflicts
RUN npm install --legacy-peer-deps

# Source
COPY web_app/frontend ./

# Build
RUN npm run build

# --- Stage 2: Serve ---
FROM nginx:alpine

# Copy Build Artifacts
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom Nginx Config (for React Router)
COPY web_app/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
