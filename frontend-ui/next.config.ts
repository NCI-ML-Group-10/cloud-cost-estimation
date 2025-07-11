/*
 * @Author: Bryan x23399937@student.ncirl.ie
 * @Date: 2025-06-20 20:12:57
 * @LastEditors: Bryan x23399937@student.ncirl.ie
 * @LastEditTime: 2025-07-11 00:12:37
 * @FilePath: /cloud-cost-estimation/frontend-ui/next.config.ts
 * @Description: 
 * 
 * Copyright (c) 2025 by Bryan Jiang, All Rights Reserved. 
 */
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'export',
};

module.exports = {
  async rewrites() {
    return [
      {
        source: '/serve/:path*',
        destination: 'http://clearml-serving.us-east-1.elasticbeanstalk.com:8080/serve/:path*',
      },
    ];
  },
};

export default nextConfig;
