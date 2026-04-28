// OpenAI-compatible adapter for self-hosted inference servers (vLLM, SGLang, etc).

import OpenAIApi from 'openai';
import { getKey, hasKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';

export class VLLM {
    static prefix = 'vllm';
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params;

        let vllm_config = {};
        vllm_config.baseURL = url || 'http://0.0.0.0:8000/v1';
        // Local servers don't enforce auth, but the OpenAI client requires a non-empty key.
        vllm_config.apiKey = hasKey('VLLM_API_KEY') ? getKey('VLLM_API_KEY') : 'local';

        this.vllm = new OpenAIApi(vllm_config);
    }

    async sendRequest(turns, systemMessage, stop_seq = '***') {
        // strictFormat must run on turns only — folding the system message in would
        // rewrite it to a user turn prefixed with "SYSTEM:", which small models follow poorly.
        let model = this.model_name || "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B";
        let messages = [{ role: 'system', content: systemMessage }].concat(strictFormat(turns));

        const pack = {
            model: model,
            messages,
            stop: stop_seq,
            ...(this.params || {})
        };

        let res = null;
        try {
            console.log('Awaiting vllm api response...')
            // console.log('Messages:', messages);
            let completion = await this.vllm.chat.completions.create(pack);
            if (completion.choices[0].finish_reason == 'length')
                throw new Error('Context length exceeded');
            console.log('Received.')
            res = completion.choices[0].message.content;
            // Strip Qwen3 / DeepSeek-R1 reasoning blocks before they reach the command parser.
            if (res && res.includes('</think>')) {
                if (!res.includes('<think>')) res = '<think>' + res;
                res = res.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
            }
        }
        catch (err) {
            if ((err.message == 'Context length exceeded' || err.code == 'context_length_exceeded') && turns.length > 1) {
                console.log('Context length exceeded, trying again with shorter context.');
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            } else {
                console.log(err);
                res = 'My brain disconnected, try again.';
            }
        }
        return res;
    }

    async saveToFile(logFile, logEntry) {
        let task_id = this.agent.task.task_id;
        console.log(task_id)
        let logDir;
        if (this.task_id === null) {
            logDir = path.join(__dirname, `../../bots/${this.agent.name}/logs`);
        } else {
            logDir = path.join(__dirname, `../../bots/${this.agent.name}/logs/${task_id}`);
        }

        await fs.mkdir(logDir, { recursive: true });

        logFile = path.join(logDir, logFile);
        await fs.appendFile(logFile, String(logEntry), 'utf-8');
    }

}