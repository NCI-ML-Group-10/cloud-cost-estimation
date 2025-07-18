import React, { useState } from 'react';
import { Form, InputNumber, Select, Button, message, Row, Col, Typography } from 'antd';

const SERVICE_OPTIONS = [
    { label: 'AI Platform', value: 'AI Platform' },
    { label: 'BigQuery', value: 'BigQuery' },
    { label: 'Cloud Armor', value: 'Cloud Armor' },
    { label: 'Cloud Build', value: 'Cloud Build' },
    { label: 'Cloud CDN', value: 'Cloud CDN' },
    { label: 'Cloud Data Fusion', value: 'Cloud Data Fusion' },
    { label: 'Cloud Dataproc', value: 'Cloud Dataproc' },
    { label: 'Cloud Endpoints', value: 'Cloud Endpoints' },
    { label: 'Cloud Functions', value: 'Cloud Functions' },
    { label: 'Cloud Interconnect', value: 'Cloud Interconnect' },
    { label: 'Cloud Load Balancing', value: 'Cloud Load Balancing' },
    { label: 'Cloud Memorystore', value: 'Cloud Memorystore' },
    { label: 'Cloud NAT', value: 'Cloud NAT' },
    { label: 'Cloud Run', value: 'Cloud Run' },
    { label: 'Cloud SQL', value: 'Cloud SQL' },
    { label: 'Cloud Spanner', value: 'Cloud Spanner' },
    { label: 'Cloud Storage', value: 'Cloud Storage' },
    { label: 'Cloud VPC', value: 'Cloud VPC' },
    { label: 'Compute Engine', value: 'Compute Engine' },
    { label: 'Container Registry', value: 'Container Registry' },
    { label: 'Dataflow', value: 'Dataflow' },
    { label: 'Firestore', value: 'Firestore' },
    { label: 'Kubernetes Engine', value: 'Kubernetes Engine' },
    { label: 'Pub/Sub', value: 'Pub/Sub' }
];

const REGION_OPTIONS = [
    { label: 'asia-east1', value: 'asia-east1' },
    { label: 'asia-northeast1', value: 'asia-northeast1' },
    { label: 'asia-southeast1', value: 'asia-southeast1' },
    { label: 'australia-southeast1', value: 'australia-southeast1' },
    { label: 'europe-north1', value: 'europe-north1' },
    { label: 'europe-west1', value: 'europe-west1' },
    { label: 'europe-west3', value: 'europe-west3' },
    { label: 'northamerica-northeast1', value: 'northamerica-northeast1' },
    { label: 'southamerica-east1', value: 'southamerica-east1' },
    { label: 'us-central1', value: 'us-central1' },
    { label: 'us-east1', value: 'us-east1' },
    { label: 'us-west1', value: 'us-west1' }
];

const defaultValues = {
    "Service Name": 'Cloud Storage',
    "Region/Zone": 'us-west1',
    "Usage Quantity": 100.0,
    "CPU Utilization (%)": 10,
    "Memory Utilization (%)": 20,
    "Network Inbound Data (Bytes)": 102400,
    "Network Outbound Data (Bytes)": 204800,
    "Cost per Quantity ($)": 2.1,
};

// const BASE_API_URL = typeof window !== 'undefined' && window.location.hostname === 'localhost'
//     ? 'http://localhost:3000/serve/cloud_cost_predict'
//     : 'http://clearml-serving.us-east-1.elasticbeanstalk.com:8080/serve/cloud_cost_predict';
const BASE_API_URL =
    typeof window !== 'undefined'
        ? `${window.location.origin}/serve/cloud_cost_predict`
        : '/serve/cloud_cost_predict'; // SSR fallback


const CostPredictionForm: React.FC = () => {
    const [loading, setLoading] = useState(false);
    const [form] = Form.useForm();
    const [result, setResult] = useState<null | number>(null);

    const onFinish = async (values: any) => {
        setLoading(true);
        setResult(null);
        try {
            const response = await fetch(BASE_API_URL, {
                method: "POST",
                headers: {
                    "accept": "application/json",
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(values)
            });
            const res = await response.json();
            if (res?.predicted_cost_usd) {
                setResult(res.predicted_cost_usd[0]);
                message.success('预测成功！');
            } else {
                message.error('无预测结果');
            }
        } catch (err) {
            message.error('请求失败');
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            {result !== null && (
                <div style={{ textAlign: 'center', marginTop: 24, marginBottom: 12 }}>
                    <Typography.Title level={4} style={{ color: '#1677ff', marginBottom: 0 }}>
                        Prediction results (USD): <span style={{ fontSize: 22 }}>{result.toFixed(2)}</span>
                    </Typography.Title>
                </div>
            )}

            <Form
                layout="vertical"
                form={form}
                initialValues={defaultValues}
                onFinish={onFinish}
                style={{
                    background: '#fff',
                    borderRadius: 12,
                    padding: 24,
                    maxWidth: 800,
                    margin: '24px auto 0'
                }}
            >
                <Row gutter={24}>
                    <Col xs={24} sm={12}>
                        <Form.Item label="Service Name" name="Service Name" rules={[{ required: true }]}>
                            <Select options={SERVICE_OPTIONS} placeholder="Select service" />
                        </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                        <Form.Item label="Region/Zone" name="Region/Zone" rules={[{ required: true }]}>
                            <Select options={REGION_OPTIONS} placeholder="Select region" />
                        </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                        <Form.Item label="Usage Quantity" name="Usage Quantity" rules={[{ required: true }]}>
                            <InputNumber min={0} style={{ width: '100%' }} />
                        </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                        <Form.Item label="CPU Utilization (%)" name="CPU Utilization (%)" rules={[{ required: true, type: 'number', min: 0, max: 100 }]}>
                            <InputNumber min={0} max={100} style={{ width: '100%' }} />
                        </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                        <Form.Item label="Memory Utilization (%)" name="Memory Utilization (%)" rules={[{ required: true, type: 'number', min: 0, max: 100 }]}>
                            <InputNumber min={0} max={100} style={{ width: '100%' }} />
                        </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                        <Form.Item label="Network Inbound Data (Bytes)" name="Network Inbound Data (Bytes)" rules={[{ required: true }]}>
                            <InputNumber min={0} style={{ width: '100%' }} />
                        </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                        <Form.Item label="Network Outbound Data (Bytes)" name="Network Outbound Data (Bytes)" rules={[{ required: true }]}>
                            <InputNumber min={0} style={{ width: '100%' }} />
                        </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                        <Form.Item label="Cost per Quantity ($)" name="Cost per Quantity ($)" rules={[{ required: true }]}>
                            <InputNumber min={0} style={{ width: '100%' }} />
                        </Form.Item>
                    </Col>
                </Row>
                <Form.Item>
                    <Button type="primary" htmlType="submit" loading={loading} block>
                        Predict
                    </Button>
                </Form.Item>
            </Form>
        </>
    );
};

export default CostPredictionForm;
