# Director Studio (apps/web) — Next.js 15 production image.
# Build context is apps/web (see docker-compose.yml), so this file only
# sees the web app — no repo-level .dockerignore surprises.

FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Baked at build time by Next.js. The default (http://localhost:8000) is
# exactly what a browser on YOUR machine needs — the compose stack maps
# the API gateway to host port 8000. Override with --build-arg only if
# you expose the API elsewhere.
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
EXPOSE 3000
CMD ["npm", "run", "start"]
