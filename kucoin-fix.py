# docker ps -a
# docker exec -u 0 -it [container] /bin/bash
# apt-get update && apt-get install nano && nano freqtrade/exchange/common.py
# docker-compose restart kucoin_1 && docker-compose restart kucoin_2 && docker-compose restart kucoin_3 && docker-compose logs -f --tail="200"

def retrier_async(f):
    async def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return await f(*args, **kwargs)
        except TemporaryError as ex:
            logger.warning('%s() returned exception: "%s"', f.__name__, ex)
            if count > 0:
                logger.warning('retrying %s() still for %s times', f.__name__, count)
                count -= 1
                kwargs.update({'count': count})
                if isinstance(ex, DDosProtection):
                    if "kucoin" in str(ex) and "429000" in str(ex):
                        logger.warning(f"Kucoin 429 error, keeping count the same.")
                        count += 1
                    else:
                        backoff_delay = calculate_backoff(count + 1, API_RETRY_COUNT)
                        logger.info(f"Applying DDosProtection backoff delay: {backoff_delay}")
                        await asyncio.sleep(backoff_delay)
                return await wrapper(*args, **kwargs)
            else:
                logger.warning('Giving up retrying: %s()', f.__name__)
                raise ex
    return wrapper