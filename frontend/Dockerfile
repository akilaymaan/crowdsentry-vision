# CrowdSentry dashboard: built by Vite, served by nginx.
#
# The nginx config serves the static bundle and reverse-proxies /api and /ws to the
# "api" service, so the browser only ever talks to this one origin -- which is also
# why the bundle is built with an empty VITE_API_BASE_URL (same-origin).

FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# VITE_* values are baked into the bundle at build time. VITE_API_KEY must match the
# backend's API_KEY; VITE_API_BASE_URL stays empty for the same-origin proxy topology.
ARG VITE_API_BASE_URL=""
ARG VITE_API_KEY=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_API_KEY=$VITE_API_KEY
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
